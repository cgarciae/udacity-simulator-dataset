import imp
import os
import shutil
import typing as tp
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from imblearn.over_sampling import RandomOverSampler
from jax.experimental import optix

import elegy


def split(df, params):

    n_split = int(params.train_split * len(df))

    df_train = df[:n_split]
    df_test = df[n_split:]

    return df_train, df_test


def balance(df, params):

    df["original_steering"] = df["steering"]

    # is_zero = df.steering == 0
    # df_zeros = df[is_zero].sample(frac=0.25)
    # df_nonzero = df[~is_zero]
    # df = pd.concat([df_zeros, df_nonzero], axis=0)

    # bins = np.linspace(-0.9, 0.9, params.n_buckets)
    # df["bucket"] = np.digitize(df["steering"], bins)
    # df, _ = RandomOverSampler().fit_resample(df, df["bucket"])

    df["flipped"] = 0
    df_flipped = df.copy()
    df_flipped["flipped"] = 1
    df = pd.concat([df, df_flipped])

    return df


def as_single_image(df, column, params):

    df = df.copy()
    df["image_path"] = df[column]
    df.drop(columns=["center", "left", "right"], inplace=True)

    return df


def preprocess(df, params, mode, directory=None):

    if directory is not None:
        for column in ["left", "center", "right"]:
            df[column] = df[column].apply(lambda x: os.path.join(directory, x))

    if mode == "train":
        df = balance(df, params)

        df_left = as_single_image(df, "left", params)
        df_right = as_single_image(df, "right", params)
        df_center = as_single_image(df, "center", params)

        df_left["steering"] += params.steering_correction
        df_right["steering"] -= params.steering_correction

        df = pd.concat([df_center, df_left, df_right], ignore_index=True)

        df.loc[df["flipped"] == 1, "steering"] = df["steering"][df["flipped"] == 1] * -1

    else:
        df = as_single_image(df, "center", params)
        df["flipped"] = 0

    df = df.sample(frac=1)
    df["angle_cat"] = np.digitize(df["steering"], [-0.1, 0.1])

    return df


def get_dataset(df, params, mode):
    dataset = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))

    dataset = dataset.repeat()

    # def filter_fn(row):
    #     return tf.math.not_equal(row["original_steering"], 0) | (
    #         tf.random.uniform(shape=()) < params.percent_zero
    #     )

    # dataset = dataset.filter(filter_fn)

    def preprocess_image(x):
        x = x[params.crop_up : -params.crop_down, :, :]
        x = cv2.resize(x, tuple(params.image_size[::-1]))
        # x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        x = x.astype(np.float32) / 255.0

        return x

    def load_row(row):
        image = tf.io.read_file(row["image_path"])
        image = tf.io.decode_image(image)

        if tf.equal(row["flipped"], 1):
            image = tf.image.flip_left_right(image)

        image = tf.numpy_function(preprocess_image, [image], tf.float32)
        image.set_shape((params.image_size[0], params.image_size[1], 3))
        # steering = row["steering"] + tf.random.normal(
        #     shape=(), stddev=params.steering_std
        # )

        # if mode == "train":
        #     if tf.equal(row["steering"], 0):
        #         sample_weight = params.percent_zero
        #     else:
        #         sample_weight = 1.0

        #     return image, row["steering"], sample_weight

        return image, row["steering"]

    dataset = dataset.map(load_row, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(100)

    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    dataset = dataset.as_numpy_iterator()

    return dataset


def get_simclr(params) -> tf.keras.Model:
    hub_url = "models/ResNet50_1x/hub"
    embed = hub.KerasLayer(hub_url, tags={"train"})

    model = tf.keras.Sequential(
        [
            embed,
            # tf.keras.layers.Linear(16, activation="relu"),
            tf.keras.layers.Linear(1, name="steering", use_bias=False),
        ],
        name="simclr_linear",
    )
    model.build((None, 224, 224, 3))
    # model.summary()
    return model


class MixtureModule(elegy.Module):
    def __init__(
        self,
        k: int,
        expert_fn: tp.Callable[[], elegy.Module],
        gating_fn: tp.Callable[[], elegy.Module],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.expert_fn = expert_fn
        self.gating_fn = gating_fn

    def call(self, x):

        y: np.ndarray = jnp.stack(
            [self.expert_fn()(x) for _ in range(self.k)], axis=1,
        )

        probs = self.gating_fn()(x)

        return y, probs


def safe_log(x):
    return jnp.log(jnp.maximum(x, 1e-6))


class MixtureLoss(elegy.Loss):
    def call(self, y_true, y_pred):
        y, probs = y_pred

        # preds = jnp.einsum("...i, ...j -> ...", probs, y[..., 0])
        # return elegy.losses.MeanSquaredError()(y_true, preds)

        # return -safe_log(
        #     jnp.sum(
        #         probs
        #         * jax.scipy.stats.norm.pdf(y_true[..., None], loc=y[..., 0], scale=1.0),
        #         axis=1,
        #     ),
        # )

        components_loss = -jax.scipy.stats.norm.logpdf(
            y_true[..., None], loc=y[..., 0], scale=1.0
        )

        mixture_loss = jnp.min(components_loss, axis=1)
        indexes = jnp.argmin(components_loss, axis=1)

        cce = elegy.losses.sparse_categorical_crossentropy(indexes, probs)

        return dict(assignments=cce, mixture=mixture_loss)


class MixtureMetrics(elegy.metrics.SparseCategoricalAccuracy):
    def call(self, y_true, y_pred):
        y, probs = y_pred

        components_loss = -jax.scipy.stats.norm.logpdf(
            y_true[..., None], loc=y[..., 0], scale=1.0
        )
        # components_loss = jnp.square(y_true - y[..., 0])

        mixture_loss = jnp.min(components_loss, axis=1)
        indexes = jnp.argmin(components_loss, axis=1)

        return super().call(indexes, probs)


def get_model(params, eager) -> elegy.Model:

    module = elegy.nn.Sequential(
        lambda: [
            elegy.nn.Conv2D(24, [5, 5], stride=2, padding="valid"),
            elegy.nn.BatchNormalization(),
            jax.nn.relu,
            elegy.nn.Conv2D(36, [5, 5], stride=2, padding="valid"),
            elegy.nn.BatchNormalization(),
            jax.nn.relu,
            elegy.nn.Conv2D(48, [5, 5], stride=2, padding="valid"),
            elegy.nn.BatchNormalization(),
            jax.nn.relu,
            elegy.nn.Conv2D(64, [3, 3], padding="valid"),
            elegy.nn.BatchNormalization(),
            jax.nn.relu,
            elegy.nn.Conv2D(64, [3, 3], padding="valid"),
            elegy.nn.BatchNormalization(),
            jax.nn.relu,
            # elegy.nn.GlobalAveragePooling2D(),
            elegy.nn.Flatten(),
            # elegy.nn.Dropout(0.4),
            elegy.nn.Linear(100),
            elegy.nn.BatchNormalization(),
            jax.nn.relu,
            MixtureModule(
                k=5,
                expert_fn=lambda: elegy.nn.Sequential(
                    lambda: [
                        elegy.nn.Linear(10),
                        elegy.nn.BatchNormalization(),
                        jax.nn.relu,
                        elegy.nn.Linear(1),
                    ]
                ),
                gating_fn=lambda: elegy.nn.Sequential(
                    lambda: [
                        elegy.nn.Linear(10),
                        elegy.nn.BatchNormalization(),
                        jax.nn.relu,
                        elegy.nn.Linear(5),
                        jax.nn.softmax,
                    ]
                ),
            ),
        ],
        name="pilot-net-elegy",
    )

    return elegy.Model(
        module,
        loss=MixtureLoss(),
        metrics=MixtureMetrics(),
        optimizer=optix.adam(params.lr),
        run_eagerly=eager,
    )

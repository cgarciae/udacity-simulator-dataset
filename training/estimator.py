import tensorflow as tf
import tensorflow_addons as tfa
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from pathlib import Path
import typing as tp
import numpy as np
import shutil
import cv2


def split(df, params):

    n_split = int(params.train_split * len(df))

    df_train = df[:n_split]
    df_test = df[n_split:]

    return df_train, df_test


def balance(df, params):

    df["original_steering"] = df["steering"]

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


def preprocess(df, params, mode):

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

    return df


def get_dataset(df, params, mode):
    dataset = tf.data.Dataset.from_tensor_slices(df.to_dict(orient="list"))

    if mode == "train":
        dataset = dataset.repeat()

        # def filter_fn(row):
        #     return tf.math.not_equal(row["original_steering"], 0) | (
        #         tf.random.uniform(shape=()) < params.percent_zero
        #     )

        # dataset = dataset.filter(filter_fn)

    def preprocess_image(x):
        x = x[params.crop_up : -params.crop_down, :, :]
        x = cv2.resize(x, tuple(params.image_size[::-1]))
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0

        return x

    def load_row(row):
        image = tf.io.read_file(row["image_path"])
        image = tf.io.decode_image(image)

        if tf.equal(row["flipped"], 1):
            image = tf.image.flip_left_right(image)

        image = tf.numpy_function(preprocess_image, [image], tf.float32)
        image.set_shape((params.image_size[0], params.image_size[1], 3))
        steering = row["steering"] + tf.random.normal(
            shape=(), stddev=params.steering_std
        )

        if mode == "train":
            if tf.equal(row["original_steering"], 0):
                sample_weight = params.percent_zero
            else:
                sample_weight = 1.0

            return image, row["steering"], sample_weight
        else:
            return image, row["steering"]

    dataset = dataset.map(load_row, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(100)

    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def get_model(params) -> tf.keras.Model:

    x = inputs = tf.keras.Input(
        shape=(params.image_size[0], params.image_size[1], 3), name="image"
    )

    x = tf.keras.layers.Conv2D(24, [5, 5], strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(36, [5, 5], strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(48, [5, 5], strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(64, [3, 3])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Conv2D(64, [3, 3])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.reshape(x, [-1, np.prod(x.shape[1:-1]), x.shape[-1]])

    x = AddPositionalEmbeddings()(x)
    # x = SelfAttentionBlock(32, head_size=16, num_heads=6, dropout=0.2)(x)
    x = SelfAttentionBlock(32, head_size=16, num_heads=12, dropout=0.2)(x)

    x = AddPositionalEmbeddings()(x)
    x = AttentionPooling(32, n_queries=5, head_size=16, num_heads=12, dropout=0.2)(x)

    # x = x[:, 0]
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Dense(1, name="steering")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="nvidia_net")

    return model


class AddPositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embeddings: tp.Optional[tf.Variable] = None

    def build(self, input_shape):

        input_shape = list(input_shape)

        self.embeddings = self.add_weight(
            name="key_kernel", shape=[1] + input_shape[1:]
        )

        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.embeddings


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
        head_size: int = 16,
        num_heads: int = 3,
        dropout: float = 0.0,
        activation: tp.Union["str", tp.Callable] = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.mha = tfa.layers.MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, dropout=dropout
        )
        self.dense = tf.keras.layers.Dense(output_size, activation=activation)

    def call(self, inputs):

        x = self.mha([inputs, inputs])
        x = self.dense(x)

        return x


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
        n_queries: int,
        head_size: int = 16,
        num_heads: int = 3,
        dropout: float = 0.0,
        activation: tp.Union["str", tp.Callable] = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_queries = n_queries
        self.mha = tfa.layers.MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, dropout=dropout
        )
        self.dense = tf.keras.layers.Dense(output_size, activation=activation)
        self.query: tp.Optional[tf.Variable] = None

    def build(self, input_shape):

        num_features = input_shape[-1]

        self.query = self.add_weight(
            name="key_kernel", shape=[1, self.n_queries, num_features]
        )

        super().build(input_shape)

    def call(self, inputs):

        query = tf.tile(
            self.query, [tf.shape(inputs)[0]] + [1] * (len(inputs.shape) - 1)
        )

        x = self.mha([query, inputs])
        x = self.dense(x)

        return x

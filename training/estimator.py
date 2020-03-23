import tensorflow as tf
import tensorflow_addons as tfa
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from pathlib import Path
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

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Dense(50)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = tf.keras.layers.Dense(1, name="steering")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="nvidia_net")

    return model


class AttentionModule(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)

        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):

        self.attention_layer = tfa.layers.MultiHeadAttention(
            self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )
        self.layer_normalization_1 = tf.keras.layers.LayerNormalization()

        self.linear = tf.keras.layers.Dense(input_shape[-1])
        self.layer_normalization_2 = tf.keras.layers.LayerNormalization()

    def call(self, x):

        x0 = x
        x = self.attention_layer([x, x, x])
        x = x0 = self.layer_normalization_1(x + x0)
        x = self.linear(x)
        x = tf.nn.relu(x)
        x = self.layer_normalization_2(x + x0)

        return x


class AddPositionalEmbeddings(tf.keras.layers.Layer):
    def build(self, input_shape):

        self.positional_embeddings = self.add_weight(
            name="positional_embeddings",
            shape=[1, input_shape[1], input_shape[2]],
            trainable=True,
        )

    def call(self, x):

        return x + self.positional_embeddings


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, num_keys, head_size, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)

        self.num_keys = num_keys
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):

        self.keys = self.add_weight("keys", shape=[1, self.num_keys, input_shape[-1]])

        self.mha = tfa.layers.MultiHeadAttention(
            head_size=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )

    def call(self, x):
        return self.mha([self.keys, x, x])

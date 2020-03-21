import tensorflow as tf
import tensorflow_addons as tfa


def split(df, params):

    n_split = int(params.train_split * len(df))

    df_train = df[:n_split]
    df_test = df[n_split:]

    return df_train, df_test

def balance(df, params):
    bins = np.linspace(-0.9, 0.9, 17)
    df["bucket"] = np.digitize(df["steering"], bins)
    df, _ = RandomOverSampler().fit_resample(df, df["bucket"])

def preprocess(df, params, mode):
    return df


def get_dataset(df, params):
    return df


def get_model(params) -> tf.keras.Model:
    pass

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path


import dataget
import dicto
import pandas as pd
import typer
import tensorflow as tf
import numpy as np

# from .
import estimator


def main(
    params_path: Path = Path("training/params.yml"),
    cache: bool = False,
    viz: bool = False,
):

    params = dicto.load(params_path)

    train_cache_path = Path("cache") / "train.feather"
    test_cache_path = Path("cache") / "test.feather"

    if cache and train_cache_path.exists() and test_cache_path.exists():
        print("Using cache...")

        df_train = pd.read_feather(train_cache_path)
        df_test = pd.read_feather(test_cache_path)

    else:
        df = dataget.image.udacity_simulator().get()

        df_train, df_test = estimator.split(df, params)

        df_train = estimator.preprocess(df_train, params, "train")
        df_test = estimator.preprocess(df_test, params, "test")

        # cache data
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        train_cache_path.parent.mkdir(exist_ok=True)

        df_train.to_feather(train_cache_path)
        df_test.to_feather(test_cache_path)

    ds_train = estimator.get_dataset(df_train, params, "train")
    ds_test = estimator.get_dataset(df_test, params, "test")

    # Visualize dataset for debuggings
    if viz:
        import matplotlib.pyplot as plt

        iteraror = iter(ds_train)
        image_batch, steer_batch = next(iteraror)
        for image, steer in zip(image_batch, steer_batch):
            plt.imshow(image.numpy().astype(np.uint8))
            plt.title(f"Steering angle: {steer}")
            plt.show()

        return

    model = estimator.get_model(params)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(params.lr), loss="mse", metrics=["mae"],
    )

    model.summary()

    model.fit(
        ds_train,
        epochs=params.epochs,
        steps_per_epoch=params.steps_per_epoch,
        validation_data=ds_test,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path("summaries") / Path(model.name)), profile_batch=0
            )
        ],
    )

    # Export to saved model
    save_path = f"models/{model.name}"
    model.save(save_path)


if __name__ == "__main__":
    typer.run(main)

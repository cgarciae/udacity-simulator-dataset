import os
from pathlib import Path

import dataget
import dicto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from jax.experimental import optix

import elegy

from . import estimator

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(
    params_path: Path = Path("training/params.yml"),
    cache: bool = False,
    viz: bool = False,
    debug: bool = False,
    eager: bool = False,
):
    if debug:
        import debugpy

        print("Waiting debuger....")
        debugpy.listen(("localhost", 5678))
        debugpy.wait_for_client()

    params = dicto.load(params_path)

    train_cache_path = Path("cache") / "train.feather"
    test_cache_path = Path("cache") / "test.feather"

    if cache and train_cache_path.exists() and test_cache_path.exists():
        print("Using cache...")

        df_train = pd.read_feather(train_cache_path)
        df_test = pd.read_feather(test_cache_path)

    else:
        df = dataget.image.udacity_simulator().get()
        # header = ["center", "left", "right", "steering", "throttle", "break", "speed"]
        # df = pd.read_csv(
        #     os.path.join(params["dataset"], "driving_log.csv"), names=header
        # )

        df_train, df_test = estimator.split(df, params)

        # df_train = estimator.preprocess(df_train, params, "train", params["dataset"])
        # df_test = estimator.preprocess(df_test, params, "test", params["dataset"])
        df_train = estimator.preprocess(df_train, params, "train")
        df_test = estimator.preprocess(df_test, params, "test")

        plt.hist(df_train.steering, bins=31)
        plt.show()

        # cache data
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        train_cache_path.parent.mkdir(exist_ok=True)

        df_train.to_feather(train_cache_path)
        df_test.to_feather(test_cache_path)

    ds_train = estimator.get_dataset(df_train, params, "train")
    ds_test = estimator.get_dataset(df_test, params, "test")

    # Visualize dataset for debuggings
    # if viz:

    #     iteraror = iter(ds_train)
    #     image_batch, steer_batch, weights = next(iteraror)
    #     for image, steer, weight in zip(image_batch, steer_batch, weights):
    #         plt.imshow(image.numpy())
    #         plt.title(f"Steering angle: {steer} weight {weight}")
    #         plt.show()

    model = estimator.get_model(params, eager)
    # model = estimator.get_simclr(params)

    x_sample, *rest = next(ds_train)
    model.summary(x_sample)

    model.fit(
        ds_train,
        epochs=params.epochs,
        steps_per_epoch=params.steps_per_epoch,
        validation_data=ds_test,
        validation_steps=params.validation_steps,
        callbacks=[
            elegy.callbacks.TensorBoard(
                logdir=str(Path("summaries") / model.module.name)
            )
        ],
    )

    # Export to saved model
    save_path = f"models/{model.module.name}"
    model.save(save_path)

    # print(f"{save_path=}")

    if viz:
        image_batch, steer_batch = next(ds_test)
        angles, preds = model.predict(image_batch)
        steering = np.einsum("...i, ...j -> ...", angles[..., 0], preds)
        for steering, image, steer in zip(steering, image_batch, steer_batch):
            plt.imshow(image)
            plt.title(f"True angle: {steer}, Pred angle: {steering}")
            plt.show()


if __name__ == "__main__":
    typer.run(main)

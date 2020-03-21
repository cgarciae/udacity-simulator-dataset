import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=[None, None, 3], name="image"),
        tf.keras.layers.Conv2D(32, kernel_size=3),
        tf.keras.layers.Conv2D(16, 3),
        tf.keras.layers.Conv2D(8, 3),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, name="steering"),
    ], name="image"
)

model.save("models/dummy_model")

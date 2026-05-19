import tensorflow as tf


def criar_cnn_pequena(
    input_shape=(64, 64, 3),
    num_classes=10
):

    modelo = tf.keras.Sequential([

        tf.keras.layers.Rescaling(
            1./255,
            input_shape=input_shape
        ),

        tf.keras.layers.Conv2D(
            16,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def criar_cnn_dropout(
    input_shape=(64, 64, 3),
    num_classes=10
):

    modelo = tf.keras.Sequential([

        tf.keras.layers.Rescaling(
            1./255,
            input_shape=input_shape
        ),

        tf.keras.layers.Conv2D(
            16,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo
"""Fast end-to-end validation for the project model stack.

This test deliberately uses synthetic images: the exact course dataset is not
redistributed in this repository. It verifies that the three model families
can be built, fitted, and used for nine-class predictions in a clean runtime.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


SEED = 42
NUM_CLASSES = 9


def build_mlp():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def build_cnn():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def verify_neural_model(model, features, labels):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(features, labels, epochs=1, batch_size=8, verbose=0)
    predictions = model.predict(features[:3], verbose=0)
    assert predictions.shape == (3, NUM_CLASSES)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


def main():
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    images = np.random.random((36, 28, 28, 3)).astype("float32")
    labels = np.arange(36, dtype="int64") % NUM_CLASSES

    random_forest = RandomForestClassifier(
        n_estimators=5, max_depth=3, random_state=SEED
    )
    random_forest.fit(images.mean(axis=-1).reshape(36, -1), labels)
    assert random_forest.predict(images[:3].mean(axis=-1).reshape(3, -1)).shape == (3,)

    verify_neural_model(build_mlp(), images, labels)
    verify_neural_model(build_cnn(), images, labels)
    print("Smoke test passed: Random Forest, MLP, and CNN all trained and predicted.")


if __name__ == "__main__":
    main()

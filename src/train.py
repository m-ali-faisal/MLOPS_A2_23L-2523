import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROC_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    t = params["train"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    x_train = np.load(os.path.join(PROC_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
    x_val   = np.load(os.path.join(PROC_DIR, "x_val.npy"))
    y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy"))

    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(t["dense_units"], activation="relu"),
        layers.Dropout(t["dropout_rate"]),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=t["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        verbose=2
    )

    model.save(os.path.join(MODEL_DIR, "model.h5"))
    pd.DataFrame(history.history).to_csv(
        os.path.join(MODEL_DIR, "history.csv"), index=False
    )
    print("Saved model and history.")

if __name__ == "__main__":
    main()
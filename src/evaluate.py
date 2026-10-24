import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROC_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model.h5")
METRICS_PATH = "metrics.json"
CM_PATH = "confusion_matrix.png"

def main():
    x_test = np.load(os.path.join(PROC_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"))

    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (acc={acc:.4f})")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120)
    plt.close()

    with open(METRICS_PATH, "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

    print(f"Test loss={loss:.4f}  acc={acc:.4f}")
    print(f"Wrote {METRICS_PATH} and {CM_PATH}")

if __name__ == "__main__":
    main()
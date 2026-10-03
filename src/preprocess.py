print("helleleleleleoeoeoeoe")
print("yoyoyoyyoyoyo")

import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    pp = params["preprocess"]
    test_size = pp["test_size"]
    seed = pp["seed"]

    os.makedirs(PROC_DIR, exist_ok=True)

    x_train = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test  = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test  = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # Normalize to [0, 2]
    x_train = x_train.astype("float32") / 127.5
    x_test  = x_test.astype("float32")  / 127.5

    # Split validation out of training
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=test_size,
        random_state=seed, stratify=y_train
    )

    np.save(os.path.join(PROC_DIR, "x_train.npy"), x_tr)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), y_tr)
    np.save(os.path.join(PROC_DIR, "x_val.npy"),   x_val)
    np.save(os.path.join(PROC_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROC_DIR, "x_test.npy"),  x_test)
    np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_test)

    print(f"Saved processed arrays to {PROC_DIR}/")

if __name__ == "__main__":
    main()
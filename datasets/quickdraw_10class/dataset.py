import os

import numpy as np
import torch
import torch.utils.data as data


DATAUTILS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(root, mtype):
    num_classes = 0
    class_names_path = os.path.join(root, "class_names.txt")
    if not os.path.exists(class_names_path):
        class_names_path = os.path.join(DATAUTILS_DIR, "class_names.txt")

    with open(class_names_path, "r", encoding="utf-8") as f:
        for _ in f:
            num_classes += 1

    cache_path = os.path.join(root, mtype + ".npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} doesn't exist!")

    print("*" * 50)
    print(f"Loading {mtype} dataset...")
    print("*" * 50)
    print(f"Classes number of {mtype} dataset: {num_classes}")
    print("*" * 50)
    data_cache = np.load(cache_path)
    return data_cache["data"].astype("float32"), data_cache["target"].astype("int64"), num_classes


class QD_Dataset(data.Dataset):
    def __init__(self, mtype, root="./quickdraw_data/Dataset"):
        self.data, self.target, self.num_classes = load_dataset(root, mtype)
        self.data = torch.from_numpy(self.data)
        self.target = torch.from_numpy(self.target)
        print(f"Dataset {mtype} loading done.")
        print("*" * 50 + "\\n")

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

    def get_number_classes(self):
        return self.num_classes

import glob
import os
from random import randint

import matplotlib.pyplot as plt
import numpy as np


def generate_dataset(
    rawdata_root="./quickdraw_data/Data",
    target_root="./quickdraw_data/Dataset",
    vfold_ratio=0.2,
    max_samples_per_class=5000,
    show_imgs=False,
):
    if not os.path.isdir(rawdata_root):
        os.makedirs(rawdata_root)
    if not os.path.isdir(target_root):
        os.makedirs(target_root)

    print("*" * 50)
    print("Generate dataset from npy data")
    print("*" * 50)
    all_files = glob.glob(os.path.join(rawdata_root, "*.npy"))
    print("Classes number: " + str(len(all_files)))
    print("*" * 50)

    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []
    class_samples_num = []

    for idx, file in enumerate(all_files):
        data = np.load(file)
        if data.shape[0] < max_samples_per_class:
            raise ValueError(
                f"Class file {file} has only {data.shape[0]} samples, "
                f"but max_samples_per_class={max_samples_per_class}."
            )

        indices = np.arange(0, data.shape[0])
        indices = np.random.choice(indices, max_samples_per_class, replace=False)
        data = data[indices]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, _ = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
        class_samples_num.append(str(data.shape[0]))

        print(
            str(idx + 1)
            + "/"
            + str(len(all_files))
            + "\t- "
            + class_name
            + " has been loaded. \n\t\t Totally "
            + str(data.shape[0])
            + " samples."
        )
        print("~" * 50)

    print("\n" + "*" * 50)
    print("Data loading done.")
    print("*" * 50)

    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0] * vfold_ratio)

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size : x.shape[0], :]
    y_train = y[vfold_size : y.shape[0]]

    print("x_train size: ")
    print(x_train.shape)
    print("\nx_test size: ")
    print(x_test.shape)

    if show_imgs:
        plt.figure("random images from dataset")
        plt.suptitle("random images from dataset")
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            idx = randint(0, len(x_train) - 1)
            plt.imshow(x_train[idx].reshape(28, 28))
            plt.title(class_names[int(y_train[idx].item())])
        plt.show()

    np.savez_compressed(target_root + "/train", data=x_train, target=y_train)
    np.savez_compressed(target_root + "/test", data=x_test, target=y_test)

    print("*" * 50)
    print("Great, data_cache has been saved into disk.")
    print("*" * 50)

    class_names_path = os.path.join(target_root, "class_names.txt")
    with open(class_names_path, "w", encoding="utf-8") as f:
        for i in range(len(class_names)):
            f.write(
                "class name: "
                + class_names[i]
                + "\t\tnumber of samples: "
                + class_samples_num[i]
                + "\n"
            )

    print("classes_names.txt has been saved.")
    print("*" * 50)

    return True

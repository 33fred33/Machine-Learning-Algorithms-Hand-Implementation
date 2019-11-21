#3£

from PIL import Image
import sklearn
import os
import numpy as np
import random as rd

# Use the same random seed all the time so that
# data is returned in a consistent order.
SEED = 7

TRAIN_ADDITIONS = ["1","2","3"]
TRAIN_FOLDERS = ["1", "2", "3"]
TEST_FOLDERS = ["T2","T1"]
CROPS_OF_ADDITIONS = 7

def main():
    X_train, X_test, y_train, y_test = load_pedestrian_data("./peddata/")
    print(y_train[:50])
    print(y_test[:50])
    for array in X_train, X_test, y_train, y_test:
        print(array.shape)
    print("Pedestrians in train data: ",sum(y_train))

def load_pedestrian_data(base_dir="."):
    """Returns X_train, X_test, y_train, y_test."""
    X_train, X_test = [], []
    y_train, y_test = [], []
    X_train, y_train = combine_unbalanced_train_dataset(base_dir)
    X_test, y_test = load_shuffled_folders(base_dir, TEST_FOLDERS)
    return X_train, X_test, y_train, y_test

def combine_unbalanced_train_dataset(base_dir):
    X, y = [], []
    directory = os.path.join(base_dir,"DC-ped-dataset_base/")
    for folder in TRAIN_FOLDERS:
        X_subset, y_subset = load_from_folder(directory, folder)
        X.extend(X_subset)
        y.extend(y_subset)
    directory = base_dir
    cr_root = rd.Random(SEED)
    for i in range(CROPS_OF_ADDITIONS):
        for folder in TRAIN_ADDITIONS:
            X_subset, y_subset = load_from_folder(directory, folder,root = cr_root)
            X.extend(X_subset)
            y.extend(y_subset)
    shuffle_both_in_same_way(X, y)
    return np.array(X), np.array(y)

def load_shuffled_folders(base_dir, folders):
    X, y = [], []
    directory = os.path.join(base_dir,"DC-ped-dataset_base/")
    for folder in folders:
        X_subset, y_subset = load_from_folder(directory, folder)
        X.extend(X_subset)
        y.extend(y_subset)
    shuffle_both_in_same_way(X, y)
    return np.array(X), np.array(y)

def load_from_folder(base_dir, name,root = None):
    ped, non_ped = [], []
    if "DC-ped-dataset_base/" in base_dir:
        non_ped = read_images(os.path.join(base_dir, f"{name}/non-ped_examples"))
        if "T" not in name:
            non_ped.extend(read_images(os.path.join(base_dir, f"{name}/non-ped_examples"),flipped = True))
        ped = read_images(os.path.join(base_dir, f"{name}/ped_examples"))
    else:
        non_ped = read_images(os.path.join(base_dir, f"{name}/add_non-ped_images"),False,root)
        non_ped.extend(read_images(os.path.join(base_dir, f"{name}/add_non-ped_images"),True,root))
        # add unbalanced non-pedestrian data
    return non_ped + ped, [0] * len(non_ped) + [1] * len(ped)

def read_images(folder, flipped=False,root = None):
    return [read_image(os.path.join(folder, filename), flipped,root) for filename in os.listdir(folder)]

def read_image(path, flipped=False,root=None):
    if flipped:
        picture = Image.open(path).transpose(Image.FLIP_LEFT_RIGHT)
    else:
        picture = Image.open(path)
    if "add_non-ped_images" in path:
        pic_size = picture.size
        w = root.randint(0,pic_size[0]-18)
        h = root.randint(0,pic_size[1]-36)
        picture = picture.crop((w, h, w+18, h+36))
    return np.frombuffer(picture.tobytes(), dtype=np.ubyte)

def shuffle_both_in_same_way(X, y):
    # By using the same seed for both shuffles, they should maintain the same order.
    shuffle_with_seed(X, SEED)
    shuffle_with_seed(y, SEED)

def shuffle_with_seed(array, seed):
    r = rd.Random(seed)
    r.shuffle(array)

if __name__ == "__main__":
    main()

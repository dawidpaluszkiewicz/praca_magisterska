import os
import shutil
import random
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Think about adding this as  parameters
random.seed(0)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def get_data_generator(dataset: str, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join("..", "data", "skin_cancer", "dataset", dataset),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
    )

    return train_generator


def read_data():
    print(
        len(
            os.listdir(
                os.path.join("..", "..", "data", "brain_tumor", "dataset", "train")
            )
        )
    )
    print(
        len(
            os.listdir(
                os.path.join("..", "..", "data", "brain_tumor", "dataset", "test")
            )
        )
    )
    print(
        len(
            os.listdir(
                os.path.join("..", "..", "data", "brain_tumor", "dataset", "val")
            )
        )
    )


def prepare_data_for_class(class_name):
    # files = os.listdir(os.path.join('..', '..', 'data', 'skin_cancer', 'images'))
    df = pd.read_csv(os.path.join("..", "..", "data", "skin_cancer", "GroundTruth.csv"))
    print(len(df))
    df = df[df[class_name] == 1]
    print(df.head())
    files = df.image.to_list()
    random.shuffle(files)

    num_of_files = len(files)

    split_train = int(num_of_files * 0.8)
    split_test = int(num_of_files * 0.9)

    train = files[:split_train]
    test = files[split_train:split_test]
    val = files[split_test:]

    print(f"train: {len(train)} files")
    print(f"train: {len(test)} files")
    print(f"train: {len(val)} files")
    #
    # # add class folders for train/test/val
    #

    # print(os.listdir(os.path.join(
    #                 "..", "..", "data", "skin_cancer", "images" )))
    train_counter = 0
    for file in train:
        # new_file_name = file.replace('(', '').replace(')', '').replace('  ', ' ').replace(' ', '_').lower()
        try:
            # print(file)
            shutil.copy(
                os.path.join(
                    "..", "..", "data", "skin_cancer", "images", file.strip() + ".jpg"
                ),
                os.path.join(
                    "..",
                    "..",
                    "data",
                    "skin_cancer",
                    "dataset",
                    "train",
                    class_name,
                    file + ".jpg",
                ),
            )
        except Exception as e:
            train_counter += 1

    print(class_name, "failed to find", train_counter)

    test_counter = 0
    for file in test:
        # new_file_name = file.replace('(', '').replace(')', '').replace('  ', ' ').replace(' ', '_').lower()
        try:
            # print(file)
            shutil.copy(
                os.path.join(
                    "..", "..", "data", "skin_cancer", "images", file.strip() + ".jpg"
                ),
                os.path.join(
                    "..",
                    "..",
                    "data",
                    "skin_cancer",
                    "dataset",
                    "test",
                    class_name,
                    file + ".jpg",
                ),
            )
        except Exception as e:
            test_counter += 1

    print(class_name, "failed to find", test_counter)

    val_counter = 0
    for file in val:
        # new_file_name = file.replace('(', '').replace(')', '').replace('  ', ' ').replace(' ', '_').lower()
        try:
            # print(file)
            shutil.copy(
                os.path.join(
                    "..", "..", "data", "skin_cancer", "images", file.strip() + ".jpg"
                ),
                os.path.join(
                    "..",
                    "..",
                    "data",
                    "skin_cancer",
                    "dataset",
                    "val",
                    class_name,
                    file + ".jpg",
                ),
            )
        except Exception as e:
            val_counter += 1

    print(class_name, "failed to find", val_counter)


def prepare_dataset():
    prepare_data_for_class("MEL")
    prepare_data_for_class("NV")
    prepare_data_for_class("BCC")
    prepare_data_for_class("AKIEC")
    prepare_data_for_class("BKL")
    prepare_data_for_class("DF")
    prepare_data_for_class("VASC")
    # MEL
    # NV
    # BCC
    # AKIEC
    # BKL
    # DF
    # VASC
    pass


if __name__ == "__main__":
    prepare_dataset()

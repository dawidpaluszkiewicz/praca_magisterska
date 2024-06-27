import os
import shutil
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Think about adding this as  parameters
random.seed(123)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def get_data_generator(dataset: str, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join("..", "data", "chest_xray", "dataset", dataset),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    return train_generator


def read_data():
    print(
        len(
            os.listdir(
                os.path.join("..", "..", "data", "chest_xray", "dataset", "train")
            )
        )
    )
    print(
        len(
            os.listdir(
                os.path.join("..", "..", "data", "chest_xray", "dataset", "test")
            )
        )
    )
    print(
        len(
            os.listdir(os.path.join("..", "..", "data", "chest_xray", "dataset", "val"))
        )
    )


def prepare_data_for_class(class_name):
    files = os.listdir(os.path.join("..", "..", "data", "brain_tumor", class_name))
    num_of_files = len(files)

    random.shuffle(files)

    split_train = int(num_of_files * 0.8)
    split_test = int(num_of_files * 0.9)

    train = files[:split_train]
    test = files[split_train:split_test]
    val = files[split_test:]

    print(f"train: {len(train)} files")
    print(f"train: {len(test)} files")
    print(f"train: {len(val)} files")

    # add class folders for train/test/val

    for file in train:
        new_file_name = (
            file.replace("(", "")
            .replace(")", "")
            .replace("  ", " ")
            .replace(" ", "_")
            .lower()
        )
        shutil.copy(
            os.path.join("..", "..", "data", "brain_tumor", class_name, file),
            os.path.join(
                "..",
                "..",
                "data",
                "brain_tumor",
                "dataset",
                "train",
                class_name,
                new_file_name,
            ),
        )

    for file in test:
        new_file_name = (
            file.replace("(", "")
            .replace(")", "")
            .replace("  ", " ")
            .replace(" ", "_")
            .lower()
        )
        shutil.copy(
            os.path.join("..", "..", "data", "brain_tumor", class_name, file),
            os.path.join(
                "..",
                "..",
                "data",
                "brain_tumor",
                "dataset",
                "test",
                class_name,
                new_file_name,
            ),
        )

    for file in val:
        new_file_name = (
            file.replace("(", "")
            .replace(")", "")
            .replace("  ", " ")
            .replace(" ", "_")
            .lower()
        )
        shutil.copy(
            os.path.join("..", "..", "data", "brain_tumor", class_name, file),
            os.path.join(
                "..",
                "..",
                "data",
                "brain_tumor",
                "dataset",
                "val",
                class_name,
                new_file_name,
            ),
        )


# def prepare_dataset():
#     prepare_data_for_class("healthy")
#     prepare_data_for_class("tumor")


# if __name__ == "__main__":
#     prepare_dataset()


import os
import shutil
import random


def move_files_by_ratio(directory_1, directory_2, ratio):
    # Get list of files in directory_1
    files_in_dir1 = os.listdir(directory_1)

    # Calculate the number of files to move
    num_files_to_move = int(len(files_in_dir1) * ratio)

    # Randomly select files to move
    files_to_move = random.sample(files_in_dir1, num_files_to_move)

    # Move the selected files to directory_2
    for file_name in files_to_move:
        source = os.path.join(directory_1, file_name)
        destination = os.path.join(directory_2, file_name)
        shutil.move(source, destination)


import os


def find_unique_extensions(directory):
    unique_extensions = set()

    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext:
                unique_extensions.add(ext.lower())

    return unique_extensions


if __name__ == "__main__":
    print(
        find_unique_extensions(
            os.path.join("..", "..", "data", "chest_xray", "dataset")
        )
    )

    # move_files_by_ratio(
    #     directory_1=os.path.join("..", "..", "data", "chest_xray", "dataset", "val", "NORMAL"),
    #     directory_2=os.path.join("..", "..", "data", "chest_xray", "dataset", "test", "NORMAL"),
    #     ratio=0.33
    # )

    # move_files_by_ratio(
    #     directory_1=os.path.join("..", "..", "data", "chest_xray", "dataset", "test", "PNEUMONIA"),
    #     directory_2=os.path.join("..", "..", "data", "chest_xray", "dataset", "val", "PNEUMONIA"),
    #     ratio=0.5
    # )

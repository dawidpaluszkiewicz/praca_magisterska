import os
import tensorflow as tf
import wandb
import time
import shutil

from wandb.keras import WandbCallback

from data_readers import mnist_reader
from data_readers import skin_cancer_data_loader
from data_readers import brain_tumor_data_reader
from data_readers import chest_pneumonia_normal_data_reader


from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    MaxPool2D,
    Flatten,
    LeakyReLU,
    BatchNormalization,
    Dropout,
    Input,
)

from tensorflow.keras.metrics import (
    Accuracy,
    AUC,
    Precision,
    Recall,
    SparseTopKCategoricalAccuracy,
    SparseCategoricalAccuracy,
)

from tensorflow.keras.optimizers import Adam

DATA_MAPPING = {
    "brain_tumor": {
        "generator": brain_tumor_data_reader.get_data_generator,
        "num_classes": 2,
    },
    "chest_pneumonia": {
        "generator": chest_pneumonia_normal_data_reader.get_data_generator,
        "num_classes": 2,
    },
    "skin_cancer": {
        "generator": skin_cancer_data_loader.get_data_generator,
        "num_classes": 7,
    },
    "mnist": {"generator": mnist_reader.get_data_generator, "num_classes": 6},
}

MODEL_MAPPING = {
    "vgg": tf.keras.applications.vgg16.VGG16,
    "resnet": tf.keras.applications.resnet50.ResNet50,
    'mobile_net': tf.keras.applications.mobilenet_v2.MobileNetV2,
    'xception': tf.keras.applications.xception.Xception,
    # 'efficientnet': tf.keras.applications.efficientnet.EfficientNetB7,
}


def prepare_model(
    model,
    input_shape=(224, 224, 3),
    optimizer="adam",
    number_of_dense_neurons=512,
    num_classes=2,
    batch_size=32
):
    pre_model = model(input_shape=input_shape, include_top=False, weights="imagenet")

    if optimizer == "adam":
        lr_denominator = batch_size / 32
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001 / lr_denominator)

    for layer in pre_model.layers:
        layer.trainable = False

    last_out = pre_model.layers[-1].output
    x = Flatten()(last_out)
    x = Dense(number_of_dense_neurons, activation="relu")(x)

    if num_classes == 2:
        x = Dense(1, activation="sigmoid")(x)
        metrics = ["accuracy", AUC(), Precision(), Recall()]
    else:
        x = Dense(num_classes, activation="softmax")(x)
        metrics = [
            SparseCategoricalAccuracy(name="acc"),
            SparseTopKCategoricalAccuracy(k=2, name="acc_top2"),
            SparseTopKCategoricalAccuracy(k=3, name="acc_top3"),
        ]  # , SparseTopKCategoricalAccuracy(k=3), Precision(), Recall()]

    model = tf.keras.Model(pre_model.input, x)
    if num_classes == 2:
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
    else:
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=metrics,
        )

    return model


def unfreeze_weights(model, num_classes):
    for layer in model.layers:
        layer.trainable = True

    optimizer = Adam(learning_rate=0.00001)
    if num_classes == 2:
        metrics = [Accuracy(), AUC(), Precision(), Recall()]
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
    else:
        metrics = [
            SparseCategoricalAccuracy(name="acc"),
            SparseTopKCategoricalAccuracy(k=2, name="acc_top2"),
            SparseTopKCategoricalAccuracy(k=3, name="acc_top3"),
        ]
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=metrics,
        )
    return model


def train_model(model, name, train, test, epochs=50):
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=3))
    callbacks.append(WandbCallback())
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(".", "models", name), save_best_only=True
        )
    )

    history = model.fit(train, validation_data=test, epochs=epochs, callbacks=callbacks)

    return model, name, history


def evaluate_model(model, data):
    ret = model.evaluate(data)
    print(ret)
    return ret


def run_experiment(
    data_set,
    model_name,
    image_size,
    optimizer,
    number_of_dense_neurons,
    batch_size,
    unfreeze_neurons=False,
):
    run = wandb.init(project=data_set)

    experiment_name = "_".join(
        [
            model_name,
            data_set,
            "_".join([str(i) for i in image_size]),
            optimizer,
            str(number_of_dense_neurons),
            str(batch_size),
            str(unfreeze_neurons)
        ]
    )

    run.name = experiment_name
    run.save()

    get_data_generator = DATA_MAPPING[data_set]["generator"]
    num_classes = DATA_MAPPING[data_set]["num_classes"]

    train = get_data_generator("train", image_size[:2], batch_size)
    test = get_data_generator("test", image_size[:2], batch_size)
    val = get_data_generator("val", image_size[:2], batch_size)

    start_time = time.time()
    model = MODEL_MAPPING[model_name]
    model = prepare_model(
        model,
        input_shape=image_size,
        optimizer=optimizer,
        number_of_dense_neurons=number_of_dense_neurons,
        num_classes=num_classes,
        batch_size=batch_size,
    )
    train_model(model, experiment_name, train, test)
    if unfreeze_neurons:
        model = tf.keras.models.load_model(os.path.join(".", "models", experiment_name))
        model = unfreeze_weights(model, num_classes)
        train_model(model, experiment_name + "_unfrozen", train, test)
        end_time = time.time()
        model = tf.keras.models.load_model(
            os.path.join(".", "models", experiment_name + "_unfrozen")
        )
        shutil.rmtree(os.path.join(".", "models", experiment_name + "_unfrozen"))
    else:
        end_time = time.time()
        model = tf.keras.models.load_model(os.path.join(".", "models", experiment_name))

    test_result = evaluate_model(model, val)

    if num_classes == 2:
        wandb.log(
            {
                "test_loss": test_result[0],
                "test_accuracy": test_result[1],
                "test_auc": test_result[2],
                "test_precision": test_result[3],
                "test_recall": test_result[4],
                "train_time": end_time - start_time,

            }
        )
    else:
        wandb.log(
            {
                "test_loss": test_result[0],
                "test_accuracy": test_result[1],
                "test_acc_top2": test_result[2],
                "test_acc_top3": test_result[3],
                "train_time": end_time - start_time,

            }
        )

    # run.finish()
    shutil.rmtree(os.path.join(".", "models", experiment_name))
    wandb.finish()

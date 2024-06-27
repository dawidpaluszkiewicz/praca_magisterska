import tensorflow as tf
from train import run_experiment, MODEL_MAPPING

DATASETS = ["brain_tumor", "mnist", "skin_cancer"]

IMAGE_SIZES = [[112, 112, 3]]  # [64, 64, 3]
OPTIMIZERS = "RMSprop"
DENSE_NEURONS = [256, 128]
BATCH_SIZES = [16, 8]
UNFREEZE_WEIGHTS = True

DEFAULT_IMAGE_SIZE = [224, 224, 3]
DEFAULT_OPTIMIZER = "adam"
DEFAULT_DENSE_NEURONS = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_UNFREEZE_WEIGHTS = False


def main():
    with tf.device("/gpu:0"):
        for model in MODEL_MAPPING.keys():
            data_set = "dementia"
            run_experiment(
                data_set=data_set,
                model_name=model,
                image_size=DEFAULT_IMAGE_SIZE,
                optimizer=DEFAULT_OPTIMIZER,
                number_of_dense_neurons=DEFAULT_DENSE_NEURONS,
                batch_size=DEFAULT_BATCH_SIZE,
                unfreeze_neurons=DEFAULT_UNFREEZE_WEIGHTS,
            )

            for image_size in IMAGE_SIZES:
                if model == "xception":
                    continue
                run_experiment(
                    data_set=data_set,
                    model_name=model,
                    image_size=image_size,
                    optimizer=DEFAULT_OPTIMIZER,
                    number_of_dense_neurons=DEFAULT_DENSE_NEURONS,
                    batch_size=DEFAULT_BATCH_SIZE,
                    unfreeze_neurons=DEFAULT_UNFREEZE_WEIGHTS,
                )

            for dense_neurons in DENSE_NEURONS:
                run_experiment(
                    data_set=data_set,
                    model_name=model,
                    image_size=DEFAULT_IMAGE_SIZE,
                    optimizer=DEFAULT_OPTIMIZER,
                    number_of_dense_neurons=dense_neurons,
                    batch_size=DEFAULT_BATCH_SIZE,
                    unfreeze_neurons=DEFAULT_UNFREEZE_WEIGHTS,
                )

            for batch_size in BATCH_SIZES:
                run_experiment(
                    data_set=data_set,
                    model_name=model,
                    image_size=DEFAULT_IMAGE_SIZE,
                    optimizer=DEFAULT_OPTIMIZER,
                    number_of_dense_neurons=DEFAULT_DENSE_NEURONS,
                    batch_size=batch_size,
                    unfreeze_neurons=DEFAULT_UNFREEZE_WEIGHTS,
                )

            run_experiment(
                data_set=data_set,
                model_name=model,
                image_size=DEFAULT_IMAGE_SIZE,
                optimizer=OPTIMIZERS,
                number_of_dense_neurons=DEFAULT_DENSE_NEURONS,
                batch_size=DEFAULT_BATCH_SIZE,
                unfreeze_neurons=DEFAULT_UNFREEZE_WEIGHTS,
            )

            run_experiment(
                data_set=data_set,
                model_name=model,
                image_size=DEFAULT_IMAGE_SIZE,
                optimizer=DEFAULT_OPTIMIZER,
                number_of_dense_neurons=DEFAULT_DENSE_NEURONS,
                batch_size=DEFAULT_BATCH_SIZE,
                unfreeze_neurons=UNFREEZE_WEIGHTS,
            )


if __name__ == "__main__":
    main()

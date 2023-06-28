import argparse
import logging
import os
from src.benchmarker import train, predict
import yaml
from src import FeatureDataset
import pickle
from datetime import datetime
from shutil import copyfile
import torch


def parse_args() -> argparse.Namespace:
    """Parse benchmarking arguments.

    Returns:
        argparse.Namespace: arguments to benchmark a model
    """
    parser = argparse.ArgumentParser(description="generate model predictions on data")
    parser.add_argument("--debug", action="store_true", help="Run with debug log level")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs/bert_config.yaml",
        help="path to config YAML file",
    )
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default="./data/",
        help="path to directory containing output from preprocessing",
    )
    parser.add_argument(
        "--train-data-file",
        type=str,
        default="train.csv",
        help="""name of the training dataset""",
    )
    parser.add_argument(
        "--test-data-file",
        type=str,
        default="test.csv",
        help="""name of the test dataset. To be used when there is no training or when
                there is no cross-validation and no train/test split""",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="directory to save predictions",
    )
    namespace = parser.parse_args()
    return namespace


def save_artifacts(
    trained_model,
    predictions,
    output_dir: str,
    config_file_path: str,
) -> None:
    """Save mode and predictions to disk."""
    timestamp_string = datetime.now().strftime("%d_%B_%Y_%H:%M:%S")
    output_dir = os.path.join(output_dir, "bert", timestamp_string)
    logging.info("Saving artifacts to {}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, "prediction.pkl")
    with open(path, "wb") as fp:
        pickle.dump(predictions, fp)

    experiment_config_file = os.path.join(output_dir, "config.yaml")
    copyfile(config_file_path, experiment_config_file)

    path = os.path.join(output_dir, "bert_clf.pt")
    torch.save(trained_model.state_dict(), path)


def run():
    """Runs cross validation and returns the train and validation results as well as the model

    Returns:
        train_list (list): List of dataframes for the k-fold train results.
        val_list (list): List of dataframes for the k-fold validation results.
        model (sklearn or TF model): Model trained in the last fold of the validation.
    """
    train_path = os.path.join(args.input_data_dir, args.train_data_file)
    test_path = os.path.join(args.input_data_dir, args.test_data_file)
    train_set = FeatureDataset(train_path, True)
    test_set = FeatureDataset(test_path, False)

    model = train(
        train_set,
        epochs=config_dict["training_params"]["epochs"],
        batch_size=config_dict["data_params"]["batch_size"],
        shuffle=config_dict["data_params"]["shuffle"],
        validation_split=config_dict["data_params"]["validation_split"],
        learning_rate=config_dict["training_params"]["learning_rate"],
        random_seed=config_dict["random_seed"],
    )
    pred = predict(model, test_set, batch_size=config_dict["data_params"]["batch_size"])

    return model, pred


if __name__ == "__main__":
    # setup configuration
    args = parse_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    config_file_path = args.config_path
    with open(config_file_path) as f:
        config_dict = yaml.safe_load(f)

    model, pred = run()

    # save artifacts
    save_artifacts(
        trained_model=model,
        predictions=pred,
        output_dir=args.output_data_dir,
        config_file_path=config_file_path,
    )

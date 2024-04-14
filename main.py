import sys
import argparse

import data_training
from data_training import training_model
from pyaml_env import parse_config, BaseConfig
from data_test import test

parser = argparse.ArgumentParser(description="MLBased WAF")
subparser = parser.add_subparsers(required=True, help="Available commands")


def training_model_all():
    test_sizes = [0.1, 0.2, 0.25, 0.3]
    for test_size in test_sizes:
        print(f"Training with test size: {test_size}")
        data_training.train(test_size, "all", "psosvm")

    # todo: add svm only algotihm
    # algorithms = ["psosvm", "svm"]
    # for alg in algorithms:
    #     print(f"Training with algorithm: {alg}")
    #     data_training.train(0.2, "all", alg)

    dataset_types = ["ecml", "csic", "all"]
    for dataset in dataset_types:
        print(f"Training with dataset: {dataset}")
        data_training.train(0.2, dataset, "psosvm")


# add argument train
parser_train = subparser.add_parser("train", help="train the model")
parser_train.add_argument("--test-size", help="specify test size for training", type=float, default=0.2)
parser_train.add_argument("--dataset", help="specify dataset to train", choices=["ecml", "csic", "all"], default="ecml")
parser_train.add_argument("--alg", help="specify algorithm to train", choices=["psosvm", "svm"], default="psosvm")
parser_train.set_defaults(func=training_model)

parser_train_all = subparser.add_parser("train_all", help="train the model with all variants")
parser_train_all.set_defaults(func=training_model_all)

# add argument test
parser_test = subparser.add_parser("test", help="test the model")
parser_test.set_defaults(func=test)

# Load config
config = BaseConfig(parse_config("config.yaml"))

if __name__ == "__main__":
    arg = sys.argv[1:]
    arg = parser.parse_args(arg)
    arg.config = config
    arg.func(arg)
    # try:
    #     arg = sys.argv[1:]
    #     arg = parser.parse_args(arg)
    #     arg.config = config
    #     arg.func(arg)
    # except Exception as e:
    #     send_notification(config.NOTIFICATION, f"Error: {e}")

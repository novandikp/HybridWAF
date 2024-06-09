import sys
import argparse
from util.Notification import send_notification
from data_training import training_model, training_all_scenario
from pyaml_env import parse_config, BaseConfig
from data_test import test_all, test_new

parser = argparse.ArgumentParser(description="MLBased WAF")
subparser = parser.add_subparsers(required=True, help="Available commands")


parser_train = subparser.add_parser("train", help="train the model")
parser_train.add_argument("--test-size", help="specify test size for training", type=float, default=0.2)
parser_train.add_argument("--dataset", help="specify dataset to train", choices=["ecml", "csic", "all"], default="ecml")
parser_train.add_argument("--alg", help="specify algorithm to train", choices=["psosvm", "svm"], default="psosvm")
parser_train.add_argument("--fitness", help="specify fitness function to train PSOSVM", choices=["accuracy", "tpr", "fpr", "fdr"], default="accuracy")
parser_train.set_defaults(func=training_model)

parser_train_all = subparser.add_parser("train_all", help="train the model with all variants")
parser_train_all.add_argument("--scenario", help="specify scenario to train", type=str, default=None)
parser_train_all.set_defaults(func=training_all_scenario)

# add argument test
parser_test = subparser.add_parser("test", help="test the model")
parser_test.set_defaults(func=test_all)

parser_test2 = subparser.add_parser("test_new", help="test the model")
parser_test2.add_argument("--variant", help="specify variant to test", type=str, default="psosvm_ecml_0.25")
parser_test2.set_defaults(func=test_new)

config = BaseConfig(parse_config("config.yaml"))

if __name__ == "__main__":
    try:
        arg = sys.argv[1:]
        arg = parser.parse_args(arg)
        arg.config = config
        arg.func(arg)
    except Exception as e:
        send_notification(config.NOTIFICATION, f"Error: {e}")

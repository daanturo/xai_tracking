#!/usr/bin/env python3


# try:
#     import cifar10
#     import gnn
#     import vgg
#     import xai
# except Exception as e:
#     print(e)

import importlib
import os
from argparse import ArgumentParser

argument_parser = ArgumentParser(description="XAI tracking")
argument_parser.add_argument("action", help="train or explain", type=str)
argument_parser.add_argument("--dataset")
argument_parser.add_argument("--model")
argument_parser.add_argument("--epochs")
argument_parser.add_argument("--history-file")
argument_parser.add_argument("--model-file")
argument_parser.add_argument("--plot-file")

arguments = argument_parser.parse_args()
dataset = arguments.dataset or "cifar10"
os.environ["DATA_SET"] = dataset
os.environ["MODEL"] = arguments.model or "VGG11"
os.environ["EPOCHS"] = arguments.epochs or "1"
os.environ["HISTORY_FILE"] = arguments.history_file or f"{dataset}_tmp.hdf5"
os.environ["MODEL_FILE"] = arguments.model_file or f"{dataset}.pt"
os.environ["PLOT_FILE"] = arguments.plot_file or f"plots.html"


def main():
    # print(arguments, [os.environ["DATA_SET"], os.environ["MODEL"], os.environ["EPOCHS"], os.environ["HISTORY_FILE"], os.environ["MODEL_FILE"],])
    lib = importlib.import_module(f"{dataset}.{arguments.action}")
    lib.main()


if __name__ == "__main__":
    main()

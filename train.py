from loader import Loader
from config import *
from model import Network
from outer import Outer
import tensorflow as tf
import numpy as np
import time


def train(loader, config):
    for train_round in range(config.train_episodes):
        input_upside_buffer, input_downside_buffer, output_upside_buffer, output_downside_buffer = loader.sample(
            config.train_buffer_size)
        network_upside.train(input_upside_buffer, output_upside_buffer)
        network_downside.train(input_downside_buffer, output_downside_buffer)

    print("train done.")


def test(loader, config):
    data = {}

    data["image_input"] = np.reshape(
        loader.test_set["input_upside_buffer"], (-1, 128, 128, 128, 1))
    data["standard_mat"] = np.reshape(
        loader.test_set["output_upside_buffer"], (-1, 16 * 7))
    print("The loss of upside:")
    network_upside.test(data)

    data["image_input"] = np.reshape(
        loader.test_set["input_downside_buffer"], (-1, 128, 128, 128, 1))
    data["standard_mat"] = np.reshape(
        loader.test_set["output_downside_buffer"], (-1, 16 * 7))
    print("The loss of downside:")
    network_downside.test(data)

if __name__ == "__main__":
    # np config
    np.set_printoptions(threshold=np.nan)

    # initialize the setting and model
    config = Config()
    loader = Loader(128, 1000, config.training_set_percent)
    loader.read_data_file()
    loader.initialize_output()
    print(loader.output)
    loader.sets_apart()

    # about the model
    network_upside = Network(config, "_upside")
    network_downside = Network(config, "_downside")

    # train & test
    if config.train:
        train(loader, config)
    if config.test:
        test(loader, config)

    # call the outer to expory a excel file
    output = Outer(config, loader, network_upside, network_downside)

    if config.output:
        output.out("tooth_result_clin3.xlsx")

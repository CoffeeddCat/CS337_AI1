from loader import Loader
from config import *
from model import Network
import tensorflow as tf
import numpy as np
import time


def train(loader, config):
    for train_round in range(config.train_episodes):
        input_upside_buffer, input_downside_buffer, output_upside_buffer, output_downside_buffer = loader.sample(
            config.train_buffer_size)
        network.train(input_upside_buffer, input_downside_buffer,
                      output_upside_buffer, output_downside_buffer)

    print("train done.")


def test(loader, config):
    data = {}
    data["image_input"] = np.reshape(
        loader.test_set["input_upside_buffer"], (-1, 128, 128, 128, 1))
    data["standard_mat"] = np.reshape(
        loader.test_set["output_upside_buffer"], (-1, 16 * 7))
    network.test(data)

if __name__ == "__main__":
    np.set_printoptions(threshold='nan')
    config = Config()
    loader = Loader(128, 1000, config.training_set_percent)
    loader.read_data_file()
    loader.initialize_output()
    loader.sets_apart()
    network = Network(config)
    train(loader, config)
    test(loader, config)

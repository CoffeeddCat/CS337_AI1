from loader import Loader
from config import *
from model import Network
import tensorflow as tf
import numpy as np
import time


def train(loader, config):
    for train_round in config.train_episodes:
        input_upside_buffer, input_downside_buffer, output_upside_buffer, output_downside_buffer = loader.sample(
            config.train_buffer_size)
        network.train()

if __name__ == "__main__":
    config = Config()
    loader = Loader()
    network = Network(config)
    train(loader, config)

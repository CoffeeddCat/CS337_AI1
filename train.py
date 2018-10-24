from loader import Loader
from config import *
from model import Network
import tensorflow as tf
import numpy as np
import time

def train(loader, config):



if __name__ == "main":
    config = Config()
    loader = Loader()
    network = Network(config)
    train(loader, config)
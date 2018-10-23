#import tensorflow as tf
import numpy as np
import os

class Loader:
    def __init__(self, voxes, max_size):
        self.voxes = voxes
        self.max_size = max_size
        self.data = np.zeros((max_size, voxes, voxes, voxes))
        self.file_list = os.listdir(r'./data')
        print(self.file_list)

    def read_file(self):
        #for test
        #self.file_list = ["test.vox"]
        for file_name, order in zip(self.file_list, range(len(self.file_list))):
            file = open('./data/' + file_name, 'r', encoding="UTF-8")
            for i in range(self.voxes):
                for j in range(self.voxes):
                    for k in range(self.voxes):
                        line = file.readline()
                        self.data[order][i][j][k] = (line=="1\n")
            print("done")
            file.close()

    def sample(self, num):
        return self.data[np.random.choice(num, self.max_size)]

    def get_data(self, index):
        return self.data[index]


loader = Loader(128, 1)
loader.read_file()
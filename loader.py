#import tensorflow as tf
import csv
import numpy as np
import os


class Loader:

    def __init__(self, voxes, max_size):
        self.voxes = voxes
        self.max_size = max_size
        self.data = np.zeros((max_size, voxes, voxes, voxes))
        self.output_data = list(csv.reader(open("parsed_data.csv")))
        # print(self.output_data)
        # self.output_data = pd.read_csv("parsed_data.csv")
        self.file_list = os.listdir(r'./data')
        print(self.file_list)

    def read_data_file(self):
        # for test
        #self.file_list = ["test.vox"]
        for file_name, order in zip(self.file_list, range(len(self.file_list))):
            file = open('./data/' + file_name, 'r', encoding="UTF-8")
            for i in range(self.voxes):
                for j in range(self.voxes):
                    for k in range(self.voxes):
                        line = file.readline()
                        self.data[order][i][j][k] = (line == "1\n")
            print("done")
            file.close()

    def initialize_output(self):
        self.output = {}
        name_temp = "!!!"
        del(self.output_data[0])
        matrix_temp = np.zeros((32, 7))
        for item in self.output_data:
            name = item[0]
            if name != name_temp:
                self.output[name] = np.zeros((32, 7))
            for i in range(2, 9):
                if int(item[1]) <= 32:
                    self.output[name][int(item[1]) - 1][i - 2] = item[i]
            name_temp = name
        # print(self.output)

    def sample(self, num):
        return self.data[np.random.choice(num, self.max_size)]

    def get_data(self, index):
        return self.data[index]

# For test.
loader = Loader(128, 1)
loader.initialize_output()
print(loader.output["leiyang"])
# loader.read_data_file()

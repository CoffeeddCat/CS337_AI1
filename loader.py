#import tensorflow as tf
import csv
import numpy as np
import os


class Loader:

    def __init__(self, voxes, max_size):
        self.voxes = voxes
        self.max_size = max_size
        self.data = {}
        self.name_array = []
        self.pool_size = 0
        self.output_data = list(csv.reader(open("parsed_data.csv")))
        # print(self.output_data)
        # self.output_data = pd.read_csv("parsed_data.csv")
        self.folder_list = os.listdir(r'./out')

    def read_data_file(self):
        # for test
        #self.file_list = ["test.vox"]
        for folder_name, order in zip(self.folder_list, range(len(self.folder_list))):
            self.name_array.append(folder_name)
            file_upside = open('./out/' + folder_name + '/' + folder_name +
                               "_initial_up.stlout.txt", 'r', encoding="UTF-8")
            file_downside = open('./out/' + folder_name + '/' + folder_name +
                                 "_initial_down.stlout.txt", 'r', encoding="UTF-8")
            self.data[
                folder_name + "_upside"] = np.zeros((self.voxes, self.voxes, self.voxes))
            self.data[
                folder_name + "_downside"] = np.zeros((self.voxes, self.voxes, self.voxes))
            for i in range(self.voxes):
                for j in range(self.voxes):
                    for k in range(self.voxes):
                        line = file_upside.readline()
                        self.data[folder_name +
                                  "_upside"][i][j][k] = (line == "1\n")

            for i in range(self.voxes):
                for j in range(self.voxes):
                    for k in range(self.voxes):
                        line = file_downside.readline()
                        self.data[folder_name +
                                  "_downside"][i][j][k] = (line == "1\n")
            file_upside.close()
            file_downside.close()
            print(folder_name + " done.")
        self.pool_size = len(self.name_array)

    def initialize_output(self):
        self.output = {}
        name_temp = "!!!"
        del(self.output_data[0])
        for item in self.output_data:
            name = item[0]
            if name != name_temp:
                self.output[name + "_upside"] = np.zeros((16, 7))
                self.output[name + "_downside"] = np.zeros((16, 7))
            for i in range(2, 9):
                if int(item[1]) <= 16:
                    self.output[
                        name + "_upside"][int(item[1]) - 1][i - 2] = item[i]
                elif int(item[1]) <= 32:
                    self.output[
                        name + "_downside"][int(item[1]) - 17][i - 2] = item[i]
            # print(self.output[name + "_upside"])
            name_temp = name
        # print(self.output)

    def sample(self, num):
        input_upside_buffer = []
        input_downside_buffer = []
        output_upside_buffer = []
        output_downside_buffer = []
        name_buffer = []
        for i in range(num):
          index = np.random.randint(0, self.pool_size-1)
          name_buffer.append(self.name_array[index])
        for name in name_buffer:
            input_upside_buffer.append(self.data[name + "_upside"])
            input_downside_buffer.append(self.data[name + "_downside"])
            output_upside_buffer.append(self.output[name + "_upside"])
            output_downside_buffer.append(
                self.output[name + "_downside"])
        return input_upside_buffer, input_downside_buffer, output_upside_buffer, output_downside_buffer

    def get_data(self, index):
        return self.data[index]

# For test.
# loader = Loader(128, 1)
# loader.read_data_file()
# loader.initialize_output()
# print(loader.sample(2))
# loader.initialize_output()
# print(loader.output["leiyang"])
# loader.read_data_file()

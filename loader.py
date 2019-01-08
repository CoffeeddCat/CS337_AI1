#import tensorflow as tf
import csv
import numpy as np
import os
import random
import copy


class Loader:

    def __init__(self, voxes, max_size, training_set_percent):

        # initialize the member
        self.voxes = voxes
        self.max_size = max_size
        self.data = {}
        self.name_array = []
        self.pool_size = 0

        # read the file
        self.output_data = list(csv.reader(open("parsedata.csv")))
        # print(self.output_data)
        # self.output_data = pd.read_csv("parsed_data.csv")

        # get the filename list
        self.folder_list = os.listdir(r'./clincheck3')

        # training set percent.
        self.training_set_percent = training_set_percent

    def read_data_file(self):
        # for test
        #self.file_list = ["test.vox"]

        # read every model file in the folder
        for folder_name, order in zip(self.folder_list, range(len(self.folder_list))):
            self.name_array.append(folder_name)
            file_upside = open('./clincheck3/' + folder_name + '/' + folder_name +
                               "_up.stlout.txt", 'r', encoding="UTF-8")
            file_downside = open('./clincheck3/' + folder_name + '/' + folder_name +
                                 "_down.stlout.txt", 'r', encoding="UTF-8")

            # initialize the data with all 0
            self.data[
                folder_name + "_upside"] = np.zeros((self.voxes, self.voxes, self.voxes))
            self.data[
                folder_name + "_downside"] = np.zeros((self.voxes, self.voxes, self.voxes))

            # filling the data
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

            # output
            print(folder_name + " done.")

        # after reading the file. count the number
        self.pool_size = len(self.name_array)

        # have a copy for divide training set and the test set
        self.name_array_copy = copy.deepcopy(self.name_array)
        self.data_copy = copy.deepcopy(self.data)

    def initialize_output(self):
        self.output = {}
        name_temp = "!!!"
        del(self.output_data[0])
        for item in self.output_data:
            name = item[0]
            if name != name_temp:
                self.output[name + "_upside"] = np.zeros((16, 7))
                self.output[name + "_downside"] = np.zeros((16, 7))
            # read the excel file to fill in the standard matrix
            for i in range(2, 8):
                if int(item[1]) <= 16:
                    self.output[
                        name + "_upside"][int(item[1]) - 1][i - 2] = item[i]
                elif int(item[1]) <= 32:
                    self.output[
                        name + "_downside"][int(item[1]) - 17][i - 2] = item[i]
            # print(self.output[name + "_upside"])
            name_temp = name
        # print(self.output)

    # def _format(self, item):
    #    item = np.reshape(item, (-1, 128, 128, 128))
    #    print(type(item))
    #    return item

    def sample(self, num):

        # sample some data from the loader
        input_upside_buffer = []
        input_downside_buffer = []
        output_upside_buffer = []
        output_downside_buffer = []
        name_buffer = []
        # print(self.pool_size)
        for i in range(num):
            index = np.random.randint(0, self.pool_size - 1)
            name_buffer.append(self.name_array[index])

        for name in name_buffer:
            input_upside_buffer.append(self.data[name + "_upside"])
            input_downside_buffer.append(self.data[name + "_downside"])
            output_upside_buffer.append(self.output[name + "_upside"])
            output_downside_buffer.append(
                self.output[name + "_downside"])

        return np.reshape(input_upside_buffer, (-1, 128, 128, 128, 1)), np.reshape(input_downside_buffer, (-1, 128, 128, 128, 1)), np.reshape(output_upside_buffer, (-1, 16 * 7)), np.reshape(output_downside_buffer, (-1, 16 * 7))

    def get_data(self, index):

        # get data by id
        return self.data[index]

    def sets_apart(self):

        # divide test set and training set
        take_out_num = self.pool_size - \
            int(self.pool_size * self.training_set_percent)
        print("test set num:", take_out_num)
        self.pool_size = self.pool_size - take_out_num
        random.shuffle(self.name_array)

        input_upside_buffer = []
        input_downside_buffer = []
        output_upside_buffer = []
        output_downside_buffer = []

        for index in range(take_out_num):
            name = self.name_array.pop()
            input_upside_buffer.append(self.data.pop(name + "_upside"))
            input_downside_buffer.append(self.data.pop(name + "_downside"))
            output_upside_buffer.append(self.output.pop(name + "_upside"))
            output_downside_buffer.append(self.output.pop(name + "_downside"))

        self.test_set = {}
        self.test_set["input_upside_buffer"] = input_upside_buffer
        self.test_set["input_downside_buffer"] = input_downside_buffer
        self.test_set["output_upside_buffer"] = output_upside_buffer
        self.test_set["output_downside_buffer"] = output_downside_buffer

    def give_all(self):

        # return all the data.
        input_upside_buffer = []
        input_downside_buffer = []
        for name in self.name_array_copy:
            input_upside_buffer.append(self.data_copy[name + "_upside"])
            input_downside_buffer.append(self.data_copy[name + "_downside"])

        return self.name_array_copy, np.reshape(input_upside_buffer, (-1, 128, 128, 128, 1)), np.reshape(input_downside_buffer, (-1, 128, 128, 128, 1))
        
# For test.
# loader = Loader(128, 1)
# loader.read_data_file()
# loader.initialize_output()
# print(loader.sample(2))
# loader.initialize_output()
# print(loader.output["leiyang"])
# loader.read_data_file()

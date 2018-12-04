import numpy as np
import xlrd
import xlwt
import openpyxl


class Outer:

    def __init__(self, config, loader, network_upside, network_downside):
        self.config = config
        self.loader = loader
        self.network_upside = network_upside
        self.network_downside = network_downside

    def initialize(file_name):
        self.file = openpyxl.Workbook()
        self.sheet = self.file.active
        self.sheet.cell(row=1, column=1, value="Name")
        self.sheet.cell(row=1, column=2, value="Tooth_id")
        self.sheet.cell(row=1, column=3, value="Translation_X")
        self.sheet.cell(row=1, column=4, value="Translation_Y")
        self.sheet.cell(row=1, column=5, value="Translation_Z")
        self.sheet.cell(row=1, column=6, value="Rotation_X")
        self.sheet.cell(row=1, column=7, value="Rotation_Y")
        self.sheet.cell(row=1, column=8, value="Rotation_Z")
        self.sheet.cell(row=1, column=9, value="Rotation_W")

    def out(self, file_name):
        self.initialize()
        name_array, input_upside, input_downside = self.loader.give_all()
        upside_result = self.network_upside.return_mat(input_upside)
        downside_result =self.network_downside.return_mat(input_downside)
        print(upside_result)
        print(downside_result)
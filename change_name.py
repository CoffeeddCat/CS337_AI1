#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

path = "out"

def rename_files(path):
    # 获取目录下所有文件，包括文件夹
    parents = os.listdir(path)
    for parent in parents:
        # 拼接一个当前文件的路径
        child = os.path.join(path, parent)
        # 判断是否为文件夹 是文件夹 递归调用函数
        if os.path.isdir(child): rename_files(child)
        else:  # 不是文件夹
            new_file_name = parent.replace(" ", "_").replace("初始状态", "initial").replace("最终状态", "final").replace("上颌", "up").replace("下颌", "down")
            # print(new_file_name)
            new_child = os.path.join(path, new_file_name)
            os.rename(child, new_child)


rename_files(path)
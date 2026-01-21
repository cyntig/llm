#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import jsonlines
import pandas as pd
from enum import Enum

class STORAGE_LEVEL(Enum):
    DISK = 1
    MEMORY = 2

class Checkpoint:
    def __init__(self, dir, file_name, storage_level=STORAGE_LEVEL.DISK):
        self.dir = dir
        self.file_name = file_name
        self.storage_level = storage_level
        self.path = os.path.join(self.dir, self.file_name)
        self.data = []
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        print("Checkpoint path: ", self.path)

    def __add__(self, line): 
        self.data.append(line)

    def get_data(self):
        return self.data
    
    def initialize(self): 
        if not os.path.exists(self.path):
            print(f"Checkpoint file not exists, path={self.path}.")
            self.data = []
            return 1
        else:
            try: 
                with jsonlines.open(self.path, 'r') as reader:
                    for line in reader:
                        self.__add__(line)
                print(f"Checkpoint initialized, len={len(self.data)}.")
                return 0
            except Exception as e: 
                self.data = []
                print(f"Checkpoint file read error, path={self.path}, error={e}.")
                return -1
    def get_continuous_index(self):
        return len(self.data)
    
    def checkpoint(self, ret):
        self.__add__(ret)
        if self.storage_level == STORAGE_LEVEL.DISK:
            self.save_checkpoint()

    def save_checkpoint(self):
        with jsonlines.open(self.path, 'w') as writer:
            writer.write_all(self.data)
        print(f"Checkpoint saved, len={len(self.data)}.")

    def remove_checkpoint(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            print(f"Checkpoint file removed, path={self.path}.")
        else:
            print(f"Checkpoint file not exists, path={self.path}.")


    def save_to_excel(self, dist_dir):
        pass
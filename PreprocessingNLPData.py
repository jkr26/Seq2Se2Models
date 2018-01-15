#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from Utility import load_file

"""
Created on Fri Dec 29 18:38:04 2017

@author: jkr
"""

class WikipediaCorpus(Dataset):
    def __init__(self, file = ""):
        """
        Args:
            file (string): Text file with 
        """
        raw_data =pd.read_csv(file, delimiter="|")
        
    
        
        
        
class MyDataset(Dataset):
    def __init__(self, file = "/home/jkr/Documents/MLData/NLPCorpora/WestburyLab.Wikipedia.Corpus.txt"):
        self.data_files = os.listdir(file)
        list.sort(self.data_files)

    def __getindex__(self, idx):
        return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)


dset = MyDataset()
loader = DataLoader(dset, num_workers=8)
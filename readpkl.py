# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:05:43 2022

@author: lily
"""

import pickle

file=open("ckpt/p_10_e21_test_results.pkl","rb")
data=pickle.load(file)
print(data)
file.close()
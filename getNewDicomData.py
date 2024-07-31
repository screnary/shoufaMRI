""" Expand origin dicom data through random volume expantion,
    Output new dicom files
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import argparse
from tqdm import tqdm
import preprocess_data as pp
import pdb


proj_root = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
data_root = os.path.join('/mnt/e/data/liuyang/', '2024shoufa')


if __name__ == '__main__':
    print("----Start----")
    test_fpath = os.path.join(data_root, '1000814任俊杰/DICOM/PA0/ST0/SE5')
    imlist = sorted(os.listdir(test_fpath),key=pp.natural_sort_key)
    pdb.set_trace()

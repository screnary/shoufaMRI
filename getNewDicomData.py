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
data_root = os.path.join(proj_root, 'Data')


if __name__ == '__main__':
    print("----Start----")
    pdb.set_trace()
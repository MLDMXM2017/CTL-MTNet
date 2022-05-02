from Config import Config
from typing import Tuple
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def get_all_data(data_path: str, mfcc_len: int = 39, class_labels: Tuple = ("angry", "fear", "happy", "neutral", "sad", "surprise"), flatten: bool = False):
    x = []
    y = []
    current_dir =  os.getcwd()
    sys.stderr.write('The current process directory is: %s\n' % current_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("Start reading folders %s\n" % directory)
        os.chdir(directory)
        for filename in tqdm(os.listdir('.')):
            if not filename.endswith('.csv'):
                continue
            filepath = os.getcwd() + '/' + filename
            feature_vector = np.loadtxt(filepath, delimiter=",", dtype = np.float32, encoding="gbk")
            x.append(feature_vector)
            y.append(i)
        sys.stderr.write("End reading folders %s\n" %directory)
        os.chdir('..')
    os.chdir(current_dir)
    return np.array(x), np.array(y)

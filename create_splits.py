import argparse
import glob
import os
import random
from random import shuffle

import pandas as pd
import numpy as np
import shutil

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.
    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function

    # adjusting file size
    val_size = 15
    test_size = 15
    train_size = 70

    # getting all the files and shuffling them
    files = [filename for filename in glob.glob(data_dir+"/processed/*.tfrecord")]    
    random.shuffle(files)

    # creating directories
    val =  data_dir + '/val'
    os.makedirs(data_dir+"/val", exist_ok=True)
    test = data_dir + '/test'
    os.makedirs(data_dir+"/test", exist_ok=True)
    train = data_dir + '/train'
    os.makedirs(data_dir+"/train", exist_ok=True)

    # moving the files according to their allocated size
    for n, fname in enumerate(files):
        if n < train_size:
            loc = 'train'
        elif n < (train_size + test_size):
            loc = 'test'
        elif n <= (train_size + test_size + val_size):
            loc = 'val'

        shutil.move(data_dir+"/processed/"+fname, data_dir+"/"+loc+"/"+fname)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
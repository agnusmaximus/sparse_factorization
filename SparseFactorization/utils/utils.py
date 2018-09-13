import sys
import os
import tensorflow as tf
import numpy as np
import random

def merge_dict(src, dst):
    """
    Merges keys in dst into src.
    This means:
    - For each key in dst:
        - If key in src, replace src's value with dst's value
        - If key not in src, add a new k,v to src dict
    """
    result_dict = dict(src)
    for k,v in dst.items():
        result_dict[k] = v
    return result_dict

def set_all_seeds(value):
    np.random.seed(0)
    random.seed(0)
    tf.set_random_seed(0)

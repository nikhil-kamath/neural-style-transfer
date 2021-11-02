import os
import tensorflow as tf

os.environ["TFHUB_MODEL_LOAD_FORMAT"] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

content_path = tf.keras.utils.get_file('me_walking.JPG')
style_path = tf.keras.utils.get_file('blue_abstract_lines.jpg')

def load_img(path_to_image):
    pass

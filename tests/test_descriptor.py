import cv2 

import pickle 
import requests

import numpy as np 
import operator as op 
import itertools as it, functools as ft

import hypothesis as hp 
import hypothesis.strategies as st 

from os import path 
from glob import glob
from datetime import timedelta 
from constants.shared import model_constants 
from descriptor.descript import Descriptor 

@hp.settings(max_examples=9)
@hp.given(
    config_filename=st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=3, max_size=10),
    weights_filename=st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=3, max_size=10)
)
def test_create_descriptor_faillure(config_filename, weights_filename):
    #setup
    current_dir = path.dirname(path.realpath(__file__))
    config_filepath = path.join(current_dir, '..', f'models/{config_filename}')
    weights_filepath = path.join(current_dir, '..', f'models/{weights_filename}') 
    #compute
    status = None 
    try:
        image_descriptor = Descriptor(config_filepath, weights_filepath)
    except Exception as e:
        status = e 
    #verification 
    assert status is not None 
    assert isinstance(status, FileNotFoundError) 

@hp.settings(max_examples=3)
@hp.given(
    thr=st.floats(min_value=0.1, max_value=0.9),
    nms=st.floats(min_value=0.1, max_value=0.5)
)
def test_create_descriptor_success(thr, nms):
    #setup
    config_filepath = model_constants['config']
    weights_filepath = model_constants['weights']
    #compute
    status = None 
    try:
        image_descriptor = Descriptor(config_filepath, weights_filepath)
    except Exception as e:
        status = e
    #verification 
    assert status is None 
    assert isinstance(image_descriptor, Descriptor)
    assert len(image_descriptor.layers) == model_constants['nb_layers']
    assert len(image_descriptor.out_layers) == 3

@hp.settings(max_examples=5)
@hp.given(image_src=st.sampled_from(
        glob( path.join(path.dirname(path.realpath(__file__)), 'images', '*.jpg')  )
    )
)
def test_decode_raw_data(image_src):
    #setup
    file_handler = open(image_src, 'rb') # access file data 
    raw_data = file_handler.read() # read the binary content 
    config_filepath = model_constants['config']
    weights_filepath = model_constants['weights']
    status = None 
    try:
        image_descriptor = Descriptor(config_filepath, weights_filepath)
    except Exception as e:
        status = e
    #compute
    decoded_image = image_descriptor.decode_raw_data(raw_data)
    #verification 
    assert status is None 
    assert len(decoded_image.shape) == 3 


@hp.settings(max_examples=32)
@hp.given(
    width=st.integers(min_value=320, max_value=640), 
    height=st.integers(min_value=240, max_value=480),
    nb_coord=st.integers(min_value=32, max_value=64) 
)
def test_if_out_of_image_shape(width, height, nb_coord):
    #setup
    x = np.random.randint(-width, width, size=(nb_coord, 1))
    y = np.random.randint(-height, height, size=(nb_coord, 1))
    w = np.random.randint(width // 2, width, size=(nb_coord, 1))
    h = np.random.randint(height // 2, height, size=(nb_coord, 1))

    matrix = np.block([x, y, w, h])  # horizontal concatenation : shape (nb_cord, 4) 
    config_filepath = model_constants['config']
    weights_filepath = model_constants['weights']
    status = None 
    try:
        image_descriptor = Descriptor(config_filepath, weights_filepath)
    except Exception as e:
        status = e
    #compute
    coordinates = image_descriptor.check_if_out_of_image_shape(matrix, width, height)
    #verification 
    assert coordinates.shape == (nb_coord, 4)  # shape conservation 
    assert np.all(coordinates[:, 2:] <= np.array([width, height])) == True # rescale wh for each object  
    assert np.all(coordinates[:, :2] >= 0) == True  # remove negative position  

@hp.settings(max_examples=5, deadline=timedelta(milliseconds=2500))
@hp.given(image_src=st.sampled_from(
        glob( path.join(path.dirname(path.realpath(__file__)), 'images', '*.jpg')  )
    )
)
def test_compute(image_src):
    #setup
    file_handler = open(image_src, 'rb') # access file data 
    raw_data = file_handler.read() # read the binary content 
    config_filepath = model_constants['config']
    weights_filepath = model_constants['weights']
    status = None 
    try:
        image_descriptor = Descriptor(config_filepath, weights_filepath)
    except Exception as e:
        status = e
    #compute
    stacked_output, scale_factor = image_descriptor.compute(raw_data)
    #verification 
    assert status is None 
    assert stacked_output[:, 5:].shape[1] == 80  # 80 coco classes dataset 
    assert np.all(scale_factor == np.array([640, 480, 640, 480]))


@hp.settings(max_examples=5, deadline=timedelta(milliseconds=5000))
@hp.given(image_src=st.sampled_from(
        glob( path.join(path.dirname(path.realpath(__file__)), 'images', '*.jpg')  )
    )
)
def test_object_detection(image_src):
    #setup
    file_handler = open(image_src, 'rb') # access file data 
    raw_data = file_handler.read() # read the binary content 
    config_filepath = model_constants['config']
    weights_filepath = model_constants['weights']
    status = None 
    try:
        image_descriptor = Descriptor(config_filepath, weights_filepath)
    except Exception as e:
        status = e
    #compute
    result = image_descriptor.detect(raw_data)
    result_keys = list(result.keys())
    result_values = list(result.values())
    result_labels = list(map(lambda idx: model_constants['labels.names'][idx], result['labels']))
    #verification
    assert status is None 
    assert result['status'] == 0  # the detection process was performed successfully 
    assert result_keys == ['status', 'coords', 'scores', 'labels']
    assert len(set(result_labels) & set(['car', 'dog', 'aeroplane'])) > 0  
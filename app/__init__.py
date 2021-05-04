from os import path 
from flask import Flask 
from logger.log import logger 
from descriptor.descript import Descriptor 

current_dir = path.dirname(path.realpath(__file__))
config_filename = path.join(current_dir, '..', 'models/yolov3.cfg')
weights_filename = path.join(current_dir, '..', 'models/yolov3.weights')

image_descriptor = None 

try:
    image_descriptor = Descriptor(config_filename, weights_filename, 0.5, 0.3)
    logger.info('the descriptor is ready')
except FileNotFoundError as contents:
    logger.error('some files are missing, please check the weight or config file')
except Exception as e:
    logger.error(e)

max_row = 1024
max_col = 1024
max_channels = 3

extentions = ['jpg', 'gif', 'png', 'ttf', 'jpeg']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = max_row * max_col * max_channels
app.config['VALID_EXTENSIONS'] =  extentions
app.config['IMAGE_DESCRIPTOR'] = image_descriptor

from app import views 

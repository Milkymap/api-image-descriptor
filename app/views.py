from os import path 
from app import app 
from flask import request, abort 
from logger.log import logger 
from werkzeug.utils import secure_filename 
from constants.shared import flask_constants
from descriptor.descript import Descriptor 

import json 
import cv2 

@app.route('/is_alive')
def is_alive():
    logger.debug('the server is alive and ready to respond')
    return json.dumps(flask_constants['is_alive'])

@app.route('/descript', methods=['POST'])
def image_description():
    file_handler = request.files['image']
    file_name = secure_filename(file_handler.filename)
    if len(file_name) > 0:
        head_part, tail_part = path.split(file_name)
        tail_name, tail_extension = tail_part.split('.')
        if tail_extension not in app.config['VALID_EXTENSIONS']:
            logger.error('this extension is not valid')
            abort(400)
    else:
        logger.error('this filename is not valid')
        abort(400)
    
    try:
        if app.config['IMAGE_DESCRIPTOR'] is not None:
            logger.debug('start object detection')
            raw_data = file_handler.read()
            output = app.config['IMAGE_DESCRIPTOR'].detect(raw_data)
            return json.dumps(output) 
        logger.debug('the descriptor is not loaded')
        logger.debug('please check the file [__init__.py]')
        return json.dumps(flask_constants['detection_failure'])
    except Exception:
        logger.debug('An exception was raised ...')
        return json.dumps(flask_constants['detection_failure'])
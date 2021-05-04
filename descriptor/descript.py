import cv2 
import numpy as np 
import operator as op 
import itertools as it, functools as ft 


from os import path 
from glob import glob 

class Descriptor:
    def __init__(self, config_filename, weights_filename, thr=0.5, nms=0.3):
        if not path.isfile(config_filename) or not path.isfile(weights_filename): 
            raise FileNotFoundError
        self.thr = thr # object detection threshold
        self.nms = nms # non maximum supression confidence 
        self.model = cv2.dnn.readNetFromDarknet(config_filename, weights_filename)
        self.layers = self.model.getLayerNames()
        self.out_layers = [self.layers[idx[0] - 1] for idx in self.model.getUnconnectedOutLayers()]
        
    def decode_raw_data(self, raw_data):
        return cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR)
    
    def make_blob(self, image):
        return cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    def compute(self, raw_data):
        image = self.decode_raw_data(raw_data)
        image_blob = self.make_blob(image)
        self.model.setInput(image_blob)
        output = self.model.forward(self.out_layers)
        stacked_output = np.vstack(output)
        imh, imw = image.shape[:2]  # discard image channels
        scale_factor = np.array([imw, imh, imw, imh]) 
        return stacked_output, scale_factor

    def check_if_out_of_image_shape(self, matrix, W, H):
        xy_part, wh_part = matrix[:, :2], matrix[:, 2:]
        df_part = np.array([W, H]) - xy_part

        xy_part = np.where(xy_part < 0, 1, xy_part)
        wh_part = np.where(wh_part > df_part, df_part, wh_part)

        return np.hstack([xy_part, wh_part])

    def process_output(self, stacked_output, scale_factor):
        coordinates, distributions = stacked_output[:, :4], stacked_output[:, 5:]
        max_distributions = np.max(distributions, axis=1)
        argmax_distributions = np.argmax(distributions, axis=1)
        retained_distributions_mask = max_distributions > self.thr
        
        selected_coordinates = coordinates[retained_distributions_mask, :] * scale_factor
        selected_max_distributions = max_distributions[retained_distributions_mask]
        selected_argmax_distributions = argmax_distributions[retained_distributions_mask]
 
        centered_selected_coordinates = np.hstack([
            selected_coordinates[:, :2] - selected_coordinates[:, 2:] / 2, 
            selected_coordinates[:, 2:]
        ])

        normalized_selected_coordinates = self.check_if_out_of_image_shape(centered_selected_coordinates, scale_factor[0], scale_factor[1])
        
        scaled_selected_coordinates = normalized_selected_coordinates.astype('uint32')
        accumulator = [scaled_selected_coordinates, selected_max_distributions,selected_argmax_distributions]
        return list(map(lambda item: item.tolist(), accumulator))

    def nms_boxes(self, accumulator):
        res = {'status': 1, 'coords': [], 'scores': [], 'labels': []}
        if len(accumulator) == 0:
            return res 
        
        index = cv2.dnn.NMSBoxes(accumulator[0], accumulator[1], self.thr, self.nms)
        flattened_index = np.ravel(index)
        for idx in flattened_index:
            res['coords'].append(accumulator[0][idx])
            res['scores'].append(accumulator[1][idx])
            res['labels'].append(accumulator[2][idx])
        res['status'] = 0 # no errors during object detection 
        return res 

    def detect(self, raw_data):
        stacked_output, scale_factor = self.compute(raw_data)
        accumulator = self.process_output(stacked_output, scale_factor)
        result = self.nms_boxes(accumulator)
        return result
from os import path 

current_dir = path.dirname(path.realpath(__file__))

flask_constants = {
    'is_alive': {
        'status': 0,
        'message': 'The server is up and ready to respond'
    },
    'detection_failure': {
        'status': 1, # error during object detection 
        'coords': [],
        'scores': [],
        'labels': []
    }
}

model_constants = {
    'labels.names': open(path.join(current_dir, 'labels.names'), 'r').read().split('\n'),
    'nb_layers': 254,
    'weights': path.join(current_dir, '..', 'models/yolov3.weights'),
    'config': path.join(current_dir, '..', 'models/yolov3.cfg')
}
 
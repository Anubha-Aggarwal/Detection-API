import base64

import cv2
import numpy as np
from flask import Flask, request
from ultralytics import YOLO

app = Flask(__name__)

# Model
model = YOLO('yolov8l.pt')


@app.route('/')
def hello_world():
    return 'This is an awesome integration API that\'s too shy'


@app.post('/upload')
def upload_base64():
    if request.method == 'POST':
        if all(key in request.form.keys() for key in ['data', 'ext']):
            # save(b64_encoded_res, request.form.get('ext'))
            return {'data': decode_detect_encode(request.form.get('data'), request.form.get('ext')),
                    'ext': request.form.get('ext')}, 200
    return {'error': 'Could not get form-data with valid keys'}, 400


def decode_detect_encode(base64_str: str, ext: str) -> str:
    im_bytes = base64.b64decode(base64_str)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    res = model(img)
    res_plotted = res[0].plot()
    jpg_img = cv2.imencode(f'.{ext}', res_plotted)
    return base64.b64encode(jpg_img[1]).decode('utf-8')


def save(s, e):
    data = base64.b64decode(s)
    with open(f'tmp.{e}', 'wb') as f:
        f.write(data)


if __name__ == '__main__':
    app.run()

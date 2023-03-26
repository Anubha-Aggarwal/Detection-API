import base64

import cv2
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Model
model = YOLO('yolov8l.pt')


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.post('/upload')
def upload_base64():
    if request.method == 'POST':
        r = request.get_json()
        print(type(r))
        print(r)
        if all(key in r.keys() for key in ['data', 'ext']):
            # save(b64_encoded_res, request.form.get('ext'))
            return {'data': decode_detect_encode(r['data'], r['ext']),
                    'ext': r['ext']}, 200
    return {'error': 'Could not get form-data with valid keys'}, 400


def decode_detect_encode(base64_str: str, ext: str) -> str:
    # Getting an encoded string
    im_str = base64.b64decode(base64_str)
    # save(im_str, ext, 'in_tmp')
    im_arr = np.frombuffer(im_str, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    res = model(img)
    res_plotted = res[0].plot()
    jpg_img = cv2.imencode(f'.{ext}', res_plotted)
    result = base64.b64encode(jpg_img[1]).decode("utf-8")
    # save(result, ext, 'out_tmp')
    return result


def save(s: str, e, f):
    st = base64.b64decode(s)
    with open(f'{f}.{e}', 'wb') as f:
        f.write(st)


if __name__ == '__main__':
    app.run()

import argparse
import base64
from datetime import datetime
from io import BytesIO
import os
import os
import shutil

from PIL import Image
import cv2
import dicto
import eventlet
import eventlet.wsgi
from flask import Flask
import numpy as np
import socketio
import tensorflow as tf
import typer

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.0
        self.error = 0.0
        self.integral = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error only if has not exploded
        if abs(self.Ki * self.integral) < 100:
            self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


sio = socketio.Server()
app = Flask(__name__)
model = None

controller = SimplePIController(0.1, 0.002)

params = dicto.load("training/params.yml")


@sio.on("telemetry")
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        image_array = image_array[params.crop_up : -params.crop_down, :, :]
        image_array = cv2.resize(image_array, tuple(params.image_size[::-1]))
        image_array = (
            cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        )

        cv2.imshow("Visualizer", (255 * image_array[..., ::-1]).astype(np.uint8))
        cv2.waitKey(1)

        preds = model(image=tf.constant(image_array[None, :, :, :]))
        steering_angle = float(preds["steering"].numpy()[0])

        throttle = controller.update(float(speed))

        # print(steering_angle, throttle)
        send_control(steering_angle, throttle)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit("manual", data={}, skip_sid=True)


@sio.on("connect")
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
        },
        skip_sid=True,
    )


def main(model_path: str, speed: float = 22):
    global app
    global model

    model_obj = tf.saved_model.load(model_path)
    model = model_obj.signatures["serving_default"]

    # print(model.structured_input_signature)
    # print(model.structured_outputs)
    # preds = model(image=tf.constant(np.random.randint(0,255,size=(1,32,32,3)).astype(np.float32)))
    # print(preds)
    # exit()

    controller.set_desired(speed)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)


if __name__ == "__main__":
    typer.run(main)

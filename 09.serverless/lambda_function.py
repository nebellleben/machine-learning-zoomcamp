#!/usr/bin/env python
# coding: utf-8

import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

onnxmodelpath = 'hair_classifier_empty.onnx'
session = ort.InferenceSession(onnxmodelpath)

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Convert to array and preprocess

def preprocess_input(img):
    x = np.array(img, dtype='float32')

    # Normalize to [0, 1]
    x = x / 255.0

    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    x = (x - mean) / std

    # Convert from (H, W, C) to (C, H, W) - channels first format
    x = x.transpose(2, 0, 1)

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    X = np.array([x])

    return X


def predict(imgurl):

    img = download_image(imgurl)
    img = prepare_image(img, (200, 200))

    X = preprocess_input(img)

    result = session.run([output_name], {input_name: X})
    predictions = result[0][0].tolist()

    return predictions


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result


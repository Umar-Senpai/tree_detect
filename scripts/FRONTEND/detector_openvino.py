#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import some common libraries
import os, cv2
import torch
from openvino.runtime import Core
import numpy as np

class detection_model_openvino:

    def __init__(self, model__path, confidence):
        # Load the network to OpenVINO Runtime.
        ie = Core()
        # model_onnx = ie.read_model(model=model__path)
        model_ir = ie.read_model(model=model__path)
        self.compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

        # Get input and output layers.
        self.output_layer_ir = self.compiled_model_ir.output(3)

    def predict(self, img):
        IMAGE_WIDTH = img.shape[1]
        IMAGE_HEIGHT = img.shape[0]
        resized_image = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        input_image = np.transpose(resized_image, (2, 0, 1))
        res_ir = self.compiled_model_ir([input_image])[self.output_layer_ir]
        return res_ir
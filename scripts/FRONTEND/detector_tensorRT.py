#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import some common libraries
import os
import sys
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart
import common
import torch
import time
from torch.nn import functional as F
import cv2
from typing import List

class detection_model_tensorRT:

    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path, confidence):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """

        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        print("HERE")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            print("HERE")
            assert runtime
            print("HERE")
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        print("HERE")
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.confidence = confidence
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def heatmaps_to_keypoints1(self, maps: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
      """
      Extract predicted keypoint locations from heatmaps.

      Args:
          maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
              each ROI and each keypoint.
          rois (Tensor): (#ROIs, 4). The box of each ROI.

      Returns:
          Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
          (x, y, logit, score) for each keypoint.

      When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
      we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
      Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
      """

      offset_x = rois[:, 0]
      offset_y = rois[:, 1]

      widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
      heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
      widths_ceil = widths.ceil()
      heights_ceil = heights.ceil()

      num_rois, num_keypoints = maps.shape[:2]
      xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 3)

      width_corrections = widths / widths_ceil
      height_corrections = heights / heights_ceil

      for i in range(num_rois):
          outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
          roi_map = F.interpolate(maps[[i]], size=outsize, mode="bicubic", align_corners=False)

          # Although semantically equivalent, `reshape` is used instead of `squeeze` due
          # to limitation during ONNX export of `squeeze` in scripting mode
          roi_map = roi_map.reshape(roi_map.shape[1:])  # keypoints x H x W

          w = roi_map.shape[2]
          pos = roi_map.view(num_keypoints, -1).argmax(1)

          x_int = pos % w
          y_int = (pos - x_int) // w

          x = (x_int.float() + 0.5) * width_corrections[i]
          y = (y_int.float() + 0.5) * height_corrections[i]

          xy_preds[i, :, 0] = x + offset_x[i]
          xy_preds[i, :, 1] = y + offset_y[i]
          xy_preds[i, :, 2] = 1

      return xy_preds

    ## Taken From https://detectron2.readthedocs.io/en/latest/_modules/detectron2/layers/wrappers.html#cat
    def cat(self, tensors: List[torch.Tensor], dim: int = 0):
        """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
        """
        assert isinstance(tensors, (list, tuple))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim)

    def infer(self, rgb_image):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        image = np.asarray(rgb_image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        common.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray([image]))

        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])

        # Process the results.
        boxes = outputs[1]
        scores = outputs[2]
        masks = outputs[4]
        confidence_len = sum(i > self.confidence for i in scores[0])
        # boxes = boxes.clip(0, 800)
        for n in range(confidence_len):
            scale_y = self.inputs[0]['shape'][3]
            scale_x = self.inputs[0]['shape'][2]
            # Append to detections
            t1 = boxes[0][n][1] * scale_x
            t2 = boxes[0][n][0] * scale_y
            t3 = boxes[0][n][3] * scale_x
            t4 = boxes[0][n][2] * scale_y
            if t2 > 1280:
              t2 = 1280
            if t4 > 1280:
              t4 = 1280
            if t1 > 800:
              t1 = 800
            if t3 > 800:
              t3 = 800
            boxes[0][n][0] = t2
            boxes[0][n][1] = t1
            boxes[0][n][2] = t4
            boxes[0][n][3] = t3
        # fix this
        boxes = boxes.clip(0)
        new_b = torch.from_numpy(boxes[:, 0:confidence_len])
        new_m = torch.from_numpy(masks[0:confidence_len])
        bboxes_flat = self.cat([b for b in new_b], dim=0)
        keypoint_results = self.heatmaps_to_keypoints1(new_m.detach(), bboxes_flat.detach())
        # keypoint_results = keypoint_results[:, :, [0, 1, 3]]
        # detections = [detections[0][0:6]]
        return keypoint_results
        # return
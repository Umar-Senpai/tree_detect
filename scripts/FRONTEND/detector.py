#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2
import torch

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

class detection_model:

    def __init__(self, model__path, confidence):

        self.logger = setup_logger(name=__name__)
        # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py        
        self.cfg = get_cfg()
        self.cfg.INPUT.MASK_FORMAT = "bitmask"
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ()
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 8
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
        self.cfg.MODEL.MASK_ON = True
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print("MODEL.DEVICE ",self.cfg.MODEL.DEVICE )
        self.cfg.MODEL.WEIGHTS = model__path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    
        # set detector
        self.predictor_synth = DefaultPredictor(self.cfg) 

    def predict(self, img):
        return self.predictor_synth(img)
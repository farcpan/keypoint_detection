import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import glob
import numpy as np
import os, json, cv2, random
import pandas as pd
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# load images
images = glob.glob("./images/*.png")
im = cv2.imread(images[0])

# Keypoint Detection
start = time.time()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

end = time.time()
print(f"Elapsed time for prediction: {end - start} [sec]")

cv2.imshow("output", v.get_image()[:, :, ::-1])
cv2.waitKey(0)

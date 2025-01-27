from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


# Initialize the configuration
cfg = get_cfg()

# Load the default configuration for segmentation
from detectron2.model_zoo import get_config_file, get_checkpoint_url

cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))

# Set the pretrained model weights
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")

# Use a single image for inference
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for predictions
cfg.MODEL.DEVICE = "cuda"  # Use "cuda" for GPU inference or "cpu" for CPU inference

output_path = "/zfs-pool/xadame44/pytorch-superpoint/segmented_image.jpg"
image_path = "/zfs-pool/xadame44/datasets/cars_train/00522.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a predictor using the configuration
predictor = DefaultPredictor(cfg)

# Perform inference on the image
outputs = predictor(image_rgb)

# Visualize the predictions
v = Visualizer(image_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Convert the visualized image to RGB format
result_image = out.get_image()[:, :, ::-1]

# Save the result to the specified output path
cv2.imwrite(output_path, result_image)
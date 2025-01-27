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
image_path = "/zfs-pool/xadame44/datasets/cars_train/00001.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_folder = '/zfs-pool/xadame44/datasets/cars_train/'
output_folder = 'masks'

# Initialize the predictor
predictor = DefaultPredictor(cfg)

for file in os.listdir(input_folder):
    img = cv2.imread(f"{input_folder}/{file}")
    outputs = predictor(img)
    instances = outputs["instances"]
    cars = instances[(instances.pred_classes == 2) | (instances.pred_classes == 7)]
    if len(cars) > 0:
        masks = cars.pred_masks
        areas = masks.sum(dim=[1, 2])
        largest_idx = areas.argmax()
        largest_mask = masks[largest_idx].cpu().numpy()
        np.savez_compressed(f"{output_folder}/{file}_mask.npz", mask=largest_mask)
    else:
        print(f"No instances of car found in {file}.")

# This is a test file for getting a single mask from an image of a car

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
image_path = "/zfs-pool/xadame44/datasets/test_car2.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a predictor using the configuration
predictor = DefaultPredictor(cfg)

# Perform inference on the image
outputs = predictor(image_rgb)

instances = outputs["instances"]
cars = instances[(instances.pred_classes == 2) | (instances.pred_classes == 7)]
if len(cars) > 0:
    masks = cars.pred_masks
    areas = masks.sum(dim=[1, 2])
    largest_idx = areas.argmax()
    largest_mask = masks[largest_idx].cpu().numpy()
    np.savez_compressed(f"{'test_car2'}_mask.npz", mask=largest_mask)

    # Save the largest mask as a binary image
#binary_mask_path = "/zfs-pool/xadame44/pytorch-superpoint/largest_car_mask.png"
#binary_mask = (largest_mask * 255).astype(np.uint8)
#cv2.imwrite(binary_mask_path, binary_mask)

# Visualize the predictions
#v = Visualizer(image_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Convert the visualized image to RGB format
#result_image = out.get_image()[:, :, ::-1]

# Save the result to the specified output path
#cv2.imwrite(output_path, result_image)
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from settings import EXPER_PATH, DATA_PATH

input_folder = 'cars_train'
output_folder = 'masks'
output_path = os.path.join(EXPER_PATH, output_folder)


os.makedirs(output_path, exist_ok = True)

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

output_path = os.path.join(EXPER_PATH, output_folder)
input_path = os.path.join(DATA_PATH, input_folder)
# Initialize the predictor
predictor = DefaultPredictor(cfg)

for file in os.listdir(input_path):
    img = cv2.imread(f"{input_path}/{file}")
    outputs = predictor(img) # Run prediction
    instances = outputs["instances"]
    cars = instances[(instances.pred_classes == 2) | (instances.pred_classes == 7)] # COCO classes, 2 = car, 7 = truck

    if len(cars) > 0:
        masks = cars.pred_masks
        areas = masks.sum(dim=[1, 2])
        largest_idx = areas.argmax()
        largest_mask = masks[largest_idx].cpu().numpy()
        np.savez_compressed(f"{output_path}/{file}_mask.npz", mask=largest_mask)

    else:
        print(f"No instances of car found in {file}.")
        os.remove(f"{input_path}/{file}")

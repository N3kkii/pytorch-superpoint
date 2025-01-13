import torch
import torch.nn as nn
import cv2
import numpy as np
import sys

# Add project path if needed (uncomment if required):
# sys.path.append("path_to_pytorch-superpoint")

from models.SuperPointNet_gauss2 import SuperPointNet_gauss2


def load_model_from_checkpoint(model_path, device):
    """Load SuperPoint model weights from a .pth.tar checkpoint that contains 'model_state_dict'."""
    model = SuperPointNet_gauss2().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # The checkpoint likely contains 'model_state_dict' instead of direct state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # If it doesn't have 'model_state_dict', then adjust accordingly
        # But given your error, this is likely the correct key.
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_image(image_path, resize=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image not found at {}".format(image_path))
    if resize is not None:
        img = cv2.resize(img, resize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_tensor = torch.from_numpy(gray.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0)
    return img, gray_tensor

def semi_to_prob(semi):
    # Convert 'semi' to probability heatmap
    dense = torch.exp(semi)
    dense_sum = torch.sum(dense, dim=1, keepdim=True)
    prob = dense / dense_sum
    prob_nodust = prob[:, :-1, :, :]  # remove dustbin
    
    B, C, Hc, Wc = prob_nodust.shape
    cell = 8
    prob_nodust = prob_nodust.view(B, cell, cell, Hc, Wc)
    prob_nodust = prob_nodust.permute(0, 3, 1, 4, 2).contiguous()
    prob_full = prob_nodust.view(B, 1, Hc*cell, Wc*cell)
    return prob_full

def extract_keypoints_from_prob(prob_map, threshold=0.015, max_num_keypoints=500):
    prob = prob_map[0,0].detach().cpu().numpy()
    keypoints = np.argwhere(prob > threshold)
    if len(keypoints) == 0:
        return [], []
    scores = prob[keypoints[:,0], keypoints[:,1]]
    idxs_sorted = np.argsort(-scores)
    if len(idxs_sorted) > max_num_keypoints:
        idxs_sorted = idxs_sorted[:max_num_keypoints]
    keypoints = keypoints[idxs_sorted]
    scores = scores[idxs_sorted]
    cv_keypoints = [cv2.KeyPoint(float(x), float(y), 1) for (y, x) in keypoints]
    return cv_keypoints, scores

if __name__ == "__main__":
    model_path = "/zfs-pool/xadame44/pytorch-superpoint/logs/superpoint_cars_scratch/checkpoints/superPointNet_100000_checkpoint.pth.tar"
    image_path = "/zfs-pool/xadame44/datasets/test_car.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model from the checkpoint
    model = load_model_from_checkpoint(model_path, device)

    # Load the image
    original_img, gray_tensor = load_image(image_path)
    gray_tensor = gray_tensor.to(device)

    # Run inference
    # After you run inference
    with torch.no_grad():
        outputs = model(gray_tensor)
        # Assuming outputs is a dict with keys 'semi' and 'desc'
        semi = outputs['semi']
        desc = outputs['desc']

    # Now semi is a tensor, so this will work
    prob_map = semi_to_prob(semi)

    # Extract keypoints
    cv_keypoints, scores = extract_keypoints_from_prob(prob_map)

    # Draw keypoints
    out_img = cv2.drawKeypoints(original_img, cv_keypoints, None, color=(0, 0, 255),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display
    #cv2.imshow("SuperPoint Keypoints", out_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Optionally save
    cv2.imwrite("detected_keypoints.jpg", out_img)

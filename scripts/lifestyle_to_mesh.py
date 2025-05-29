import argparse
import cv2
import torch
import numpy as np
import trimesh
from pathlib import Path

# Import ZoeDepth model
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.utils.misc import pil_to_batched_tensor, colorize

def load_zoedepth(device="cuda" if torch.cuda.is_available() else "cpu"):
    config = get_config("zoedepth_nk")  # nk = no KITTI fine-tuning, general model
    model = build_model(config)
    model.eval().to(device)
    return model, device

def predict_depth(model, device, image_path):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = pil_to_batched_tensor(img_rgb).to(device)

    with torch.no_grad():
        depth = model.infer(img_tensor)[0]
    
    depth_np = depth.squeeze().cpu().numpy()
    return depth_np, img

def depth_to_point_cloud(depth_map, rgb_img, scale=1.0):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    # Simple perspective projection assumptions
    z = depth_map * scale
    x = (i - w / 2) * z / w
    y = (j - h / 2) * z / h

    # Flatten and stack into point cloud
    points = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)
    colors = rgb_img.reshape(-1, 3) / 255.0

    return points, colors

def point_cloud_to_mesh(points, colors, output_path):
    # Remove invalid points (e.g., depth = 0)
    valid = ~np.isnan(points).any(axis=1)
    points = points[valid]
    colors = colors[valid]

    # Use Trimesh to build a mesh (triangulation on XY)
    cloud = trimesh.points.PointCloud(points, colors)
    mesh = cloud.to_mesh()
    
    mesh.export(str(output_path))
    print(f"Saved mesh to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate .obj mesh from lifestyle shot using ZoeDepth")
    parser.add_argument("image", type=str, help="Path to lifestyle shot")
    parser.add_argument("--output", type=str, default="output_mesh.obj", help="Output OBJ file path")
    args = parser.parse_args()

    model, device = load_zoedepth(device=None)
    depth_map, rgb_img = predict_depth(model, device, args.image)
    points, colors = depth_to_point_cloud(depth_map, rgb_img)
    point_cloud_to_mesh(points, colors, args.output)

if __name__ == "__main__":
    main()

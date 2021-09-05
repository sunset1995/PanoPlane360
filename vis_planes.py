import open3d
import argparse
import numpy as np
from imageio import imread
from utils import read_h_planes, read_v_planes
import torch

from models import models_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', required=True)
    parser.add_argument('--h_planes', required=True)
    parser.add_argument('--v_planes', required=True)
    parser.add_argument('--topview', action='store_true')
    args = parser.parse_args()

    # Read input
    v_planes = imread(args.v_planes)
    h_planes = imread(args.h_planes)
    H_, W = h_planes.shape
    cropped = (W//2 - H_) // 2
    h_planes[H_//2-10:H_//2+10] = 0
    h_planar = h_planes != 0
    v_planar = np.abs(v_planes).sum(-1) != 0
    rgb = imread(args.img)[..., :3]
    if cropped > 0:
        rgb = rgb[cropped:-cropped]

    # Planes to depth
    v2d = models_utils.vplane_2_depth(torch.FloatTensor(v_planes.transpose(2, 0, 1)[None]))[0, 0]
    h2d = models_utils.hplane_2_depth(torch.FloatTensor(h_planes[None, None]))[0, 0]
    depth = np.zeros([H_, W])
    depth[v_planar] = v2d[v_planar]
    depth[h_planar] = h2d[h_planar]
    depth = np.clip(depth, 0, 20)

    # Project to 3d
    v_grid = models_utils.v_grid(1, 1, H_, W)[0, 0].numpy()  # H_, W
    u_grid = models_utils.u_grid(1, 1, H_, W)[0, 0].numpy()  # H_, W
    zs = depth * np.sin(v_grid)
    xs = depth * np.cos(v_grid) * np.cos(u_grid)
    ys = depth * np.cos(v_grid) * np.sin(u_grid)
    pts_xyz = np.stack([xs, ys, zs], -1).reshape(-1, 3)
    pts_rgb = rgb.reshape(-1, 3) / 255

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts_xyz)
    pcd.colors = open3d.utility.Vector3dVector(pts_rgb)
    open3d.visualization.draw_geometries([
        pcd,
        open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    ])


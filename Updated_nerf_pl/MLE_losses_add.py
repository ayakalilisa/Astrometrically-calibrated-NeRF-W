import torch
import numpy as np
from utils.colmap_utils import read_cameras_text, read_images_text

def mle_loss(f, p, colmap_path, voxel_grid, voxel_size):
    def build_forward_matrix(colmap_path, voxel_grid, voxel_size):
        cameras = read_cameras_text(os.path.join(colmap_path, "cameras.txt"))
        images = read_images_text(os.path.join(colmap_path, "images.txt"))

        A_list = []

        for image_id, img in images.items():
            R = img.qvec2rotmat()
            t = img.tvec
            cam = cameras[img.camera_id]

            K = np.array([
                [cam.params[0], 0, cam.params[1]],
                [0, cam.params[0], cam.params[2]],
                [0, 0, 1]
            ])

            # Loop over sampled pixel coordinates
            for u in range(0, cam.width, 8):  # subsample every 8 px for speed
                for v in range(0, cam.height, 8):
                    # pixel to ray in camera frame
                    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
                    d_world = R.T @ d_cam
                    o_world = -R.T @ t

                    # compute intersections with voxel grid (simple raymarch)
                    weights = np.zeros(len(voxel_grid))
                    for i, voxel_center in enumerate(voxel_grid):
                        # check intersection distance / contribution
                        # e.g. using Beer–Lambert attenuation model modify if needed
                        weights[i] = np.exp(-np.linalg.norm(voxel_center - o_world) / voxel_size)
                    A_list.append(weights)

        A = np.vstack(A_list)
        return A
    # Run forward_matrix
    A = build_forward_matrix(colmap_path, voxel_grid, voxel_size)
    """
    Poisson negative log-likelihood loss
    with input parameters f: nerf predictions constructed 3d sample grid (sigma_s * c_s + sigma_t * c_t)
                          A: forward projector from nerfw
                          p: observed measurements from nerfw c_s + c_t
    """
    EPS = 1e-6
    h = torch.clamp(A @ f, min=EPS)  # predicted rates
    loss_mle = torch.sum(h - p * torch.log(h))
    return loss_mle

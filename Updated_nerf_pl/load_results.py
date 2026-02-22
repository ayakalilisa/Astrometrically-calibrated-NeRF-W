
'''
This script is for load and visualizing the results from trained nerf-w
    Load trained checkpoint epoch = 19.ckpt into NeRFSystem
    create phototourismdataset with split for test
    the test images will be looped over and render with the model
    *The output result will be saved as png and stacked into .mp4
'''
import torch
from torch.utils.data import DataLoader
import imageio
import os
from datasets.phototourism import PhototourismDataset
from system import NeRFsystem

# Load the ckpts
ckpt_path = "ckpts/flammer_nerfw/epoch=19.ckpt"
system = NeRFsystem.load_checkpoint(ckpt_path)
system.cuda().eval()


# Load dataset
dataset = PhototourismDataset(root_dir = "Flammer_img", split = 'test', img_downscale = 8)
load = Dataloader(dataset, batch_size = 1, shuffle = False)

# Output folder
out_dir = "Flammer_results"
os.makedirs(out_dir, exist_ok = True)
frames = []

# Render loop
for i, batch in enumerate(load):
    rays, rgbs, ts = batch["rays"].cuda(), batch["rgbs"].cuda(), batch["ts"].cuda()
    with torch.no_grad():
        results = system(rays, ts)

    # Convert predicted RGB
    typ = "fine" if "rgb_fine" in results else "coarse"
    pred_img = results[f"rgb_{typ}"].view(dataset.test_img_h, dataset.test_img_w, 3).cpu().numpy()
    pred_img = (pred_img * 255).astype("uint8")

    # Save PNG
    fname = os.path.join(out_dir, f"frame_{i:04d}.png")
    imageio.imwrite(fname, pred_img)
    frames.append(pred_img)

# Save MP4
imageio.mimwrite(os.path.join(out_dir, "flammer_video.mp4"), frames, fps=30)
print("Saved results to", out_dir)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import PhototourismDataset, BlenderDataset
from models.nerf import NeRF
from utils import load_ckpt
from opt import get_opts
from renderer import batched_inference
from embedder import PosEmbedding

@torch.no_grad()
def extract_single_depth(args, img_index=0):
    # ---------- Load dataset (only 1 image needed) ----------
    kwargs = {'root_dir': args.root_dir, 'split': 'test'}
    if args.dataset_name == 'blender':
        dataset = BlenderDataset(img_wh=tuple(args.img_wh), **kwargs)
    else:
        dataset = PhototourismDataset(
            root_dir=args.root_dir,
            split='train',        # load poses & rays
            img_downscale=args.img_downscale,
            use_cache=args.use_cache,
            N_vocab=args.N_vocab,
            encode_a=args.encode_a,
            encode_t=args.encode_t,
            beta_min=args.beta_min
        )

    sample = dataset[img_index]
    rays = sample['rays'].cuda()
    ts = sample['ts'].cuda()

    # ---------- Load embedding functions ----------
    emb_xyz = PosEmbedding(args.N_emb_xyz - 1, args.N_emb_xyz).cuda()
    emb_dir = PosEmbedding(args.N_emb_dir - 1, args.N_emb_dir).cuda()

    embeddings = {'xyz': emb_xyz, 'dir': emb_dir}

    # ---------- Load NeRF models ----------
    nerf_coarse = NeRF('coarse',
                       in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3).cuda()

    nerf_fine = NeRF('fine',
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     beta_min=args.beta_min).cuda()

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    # ---------- Run inference on ONE frame ----------
    results = batched_inference(
        models, embeddings,
        rays, ts,
        args.N_samples,
        args.N_importance,
        args.use_disp,
        args.chunk,
        white_back=False
    )

    w, h = sample['img_wh']
    depth = results['depth_fine'].view(h, w).cpu().numpy()

    # ---------- Save depth ----------
    out_dir = "depth_output"
    os.makedirs(out_dir, exist_ok=True)

    np.save(f"{out_dir}/depth.npy", depth)

    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    plt.imsave(f"{out_dir}/depth.png", depth_norm, cmap='magma')

    print("✓ Saved depth.npy and depth.png to:", out_dir)


if __name__ == "__main__":
    args = get_opts()
    extract_single_depth(args, img_index=0)  # pick image 0


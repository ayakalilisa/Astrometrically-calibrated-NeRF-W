import torch
import os
import numpy as np
from datasets import PhototourismDataset
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'phototourism'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'test_train'])
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    # original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', default=False, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=16,
                        help='number of embeddings for transient objects')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='minimum color variance for each ray')

    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='gif',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

from collections import defaultdict
from datasets import PhototourismDataset

def eval_phototourism(args, system):
    dataset = PhototourismDataset(
        root_dir=args.root_dir,
        split=args.split,
        img_downscale=args.img_downscale,
        use_cache=args.use_cache,
        N_vocab=args.N_vocab,
        encode_a=args.encode_a,
        encode_t=args.encode_t,
        beta_min=args.beta_min
    )

    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        rays = sample['rays'].to(system.device)
        ts = sample['ts'].to(system.device)

        with torch.no_grad():
            rendered = batched_inference(
                system.models,
                system.embeddings,
                rays,
                ts,
                args.N_samples,
                args.N_importance,
                args.use_disp,
                args.chunk,
                white_back=False  # False for dark background and True for bright background
            )

        rgb = rendered['rgb'].cpu().numpy()
        img = (rgb.reshape(sample['img_wh'][1], sample['img_wh'][0], 3) * 255).astype(np.uint8)
        results.append(img)



    # save video
    import imageio
    out_path = f"{args.ckpt_path.replace('.ckpt','')}_{args.split}.{args.video_format}"
    imageio.mimwrite(out_path, results, fps=30, quality=8)
    print(f"Saved video to {out_path}")
    print(f'RGB min,max checkpoint: {img.min(), img.max()}')

if __name__ == "__main__":
    args = get_opts()
    # --- AUTO-LOAD SETTINGS FROM CHECKPOINT ---
    ckpt_tmp = torch.load(args.ckpt_path, map_location='cpu')
    if 'hyper_parameters' in ckpt_tmp:
        hp = ckpt_tmp['hyper_parameters']
        print("Using hyperparameters from checkpoint:")
        for k in ['dataset_name', 'img_wh', 'N_emb_dir', 'N_emb_xyz', 'N_vocab',
                  'encode_t', 'encode_a', 'beta_min']:
            '''
            if k in hp:
                old_val = getattr(args, k, None)
                new_val = hp[k]
                setattr(args, k, new_val)
            '''
            for k, v in hp.items():
                # never overwrite these
                if k in {"ckpt_path", "N_vocab", "N_emb_xyz", "N_emb_dir", "encode_a", "encode_t"}:
                    continue

                if hasattr(args, k):
                    setattr(args, k, v)
            '''
            for k in hp:
                # DO NOT overwrite architectural params
                skip_params = {'N_vocab', 'N_emb_xyz', 'N_emb_dir', 'encode_a', 'encode_t'}
                if k not in skip_params and hasattr(args, k):
                    setattr(args, k, hp[k])
            for k, v in hp.items():
                print(f"{k}: {v}")
            '''
        print()



    # --- CREATE EMBEDDINGS ---
    embedding_xyz = PosEmbedding(args.N_emb_xyz - 1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir - 1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda()
        load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a

    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda()
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    # --- DATASET ---
    kwargs = {'root_dir': args.root_dir, 'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache

    dataset = dataset_dict[args.dataset_name](**kwargs)
    scene = os.path.basename(args.root_dir.strip('/'))

    if args.dataset_name == 'phototourism':

        if args.encode_a:
            embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda()
            load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
            embeddings['a'] = embedding_a
        if args.encode_t:
            embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda()
            load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
            embeddings['t'] = embedding_t


    # Modified appearance/transient embeddings for both datastypes


    nerf_coarse = NeRF('coarse',
                        in_channels_xyz=6*args.N_emb_xyz+3,
                        in_channels_dir=6*args.N_emb_dir+3).cuda()
    models = {'coarse': nerf_coarse}
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

    imgs, psnrs = [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {}
    # define testing poses and appearance index for phototourism
    if args.dataset_name == 'phototourism' and args.split == 'test':
        # define testing camera intrinsics (hard-coded, feel free to change)
        dataset.test_img_w = 512  # pick your test resolution width
        dataset.test_img_h = 512  # and height (adjust to your dataset)
        dataset.test_focal = dataset.test_img_w / 2 / np.tan(np.pi / 6)  # fov = 60 degrees
        dataset.test_K = np.array([
            [dataset.test_focal, 0, dataset.test_img_w / 2],
            [0, dataset.test_focal, dataset.test_img_h / 2],
            [0, 0, 1]
        ])

        # --- Replace scene-specific hardcode ---
        # Choose any valid image ID from your dataset as reference
        ref_id = list(dataset.poses_dict.keys())[0]  # pick the first one
        dataset.test_appearance_idx = ref_id

        N_frames = 50  # or however many frames you want in your video
        #dx = np.linspace(0, 0.02, N_frames)
        #dy = np.linspace(0, 0.0, N_frames)
        #dz = np.linspace(0, 0.2, N_frames)

        # Use a slightly bigger motion for flames
        dx = np.linspace(-0.5, 0.5, N_frames)
        dy = np.linspace(-0.2, 0.2, N_frames)
        dz = np.linspace(-0.3, 0.3, N_frames)

        dataset.poses_test = np.tile(dataset.poses_dict[ref_id], (N_frames, 1, 1))
        for i in range(N_frames):
            dataset.poses_test[i, 0, 3] += dx[i]
            dataset.poses_test[i, 1, 3] += dy[i]
            dataset.poses_test[i, 2, 3] += dz[i]

        kwargs['output_transient'] = False

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)

        if args.dataset_name == 'blender':
            w, h = args.img_wh
        else:
            w, h = sample['img_wh']

        # ----- DEPTH EXTRACTION -----
        print("RESULT KEYS:", results.keys())
        frames_to_show = {11, 12, 26, 27}

        print("RESULT KEYS:", results.keys())

        if "depth_fine" in results:
            depth = results["depth_fine"].view(h, w).cpu().numpy()

            if i in frames_to_show:
                print(f"Saving depth for eval index {i}")

                plt.figure(figsize=(6, 6))
                plt.imshow(depth, cmap="magma")
                plt.colorbar()
                plt.axis("off")
                plt.title(f"Depth map (eval index {i})")

                out_path = os.path.join(args.root_dir, f"depth_{i}.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
        else:
            print("WARNING: depth_fine missing from results")

        # --------------------------------
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    if args.dataset_name == 'blender' or \
      (args.dataset_name == 'phototourism' and args.split == 'test'):
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                        imgs, fps=30)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
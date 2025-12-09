import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from monet import Monet, MonetConfig


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=r"""
Example usage:

TODO
""")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=5, help='Batch size to use.')
    parser.add_argument("--diffusion_steps", type=int, default=10, help='Number of diffusion steps to use.')
    parser.add_argument("input", type=str, help="Single brightfield image or directory of channels to be cellpainted. When providing a directory, files must specify which image they are a part of with their prefix " 
                                                "and which channel they are with their suffix. Channels with no suffix specified will be treated as brightfield channels. "
                                                "Any not specified channels will not be used for conditioning. Expected suffixes are _brightfield_reference, _mito, _rna, _er, _dna, _agp, and _brightfield. "
                                                "_brightfield suffix is the brightfield to cellpaint. _brightfield_reference suffix is the brightfield of the reference conditioning image. "
                                                "i.e. <input>/00000.png <input>/00001.png <input>/00002.png <input>/00003.png is four separate brightfields to be cellpainted. "
                                                "<input>/00000_brightfield_reference.png <input>/00000_mito.png <input>/00000_rna.png <input>/00000_er.png <input>/00000_dna.png <input>/00000_agp.png <input>/00000_brightfield.png is a "
                                                "single image to be cellpainted with a specified reference image to condition the generation on."
                                                )
    parser.add_argument("-o", "--output", type=str, default="output", 
                                            help="Output directory. Must not exist or be empty. Generated channels and composite images will be saved to this directory with the same "
                                                "prefix as in the input files. Composite will be saved as <prefix>.png and individual channels will be saved as <prefix>_<channel_name>.png.")
    parser.add_argument("--checkpoint", type=str, default=None, required=False, help="Optional path to local checkpoint to load. If not provided, uses the base model.")

    args = parser.parse_args()

    if os.path.exists(args.output) and len(os.listdir(args.output)) > 0:
        print(f"{args.output} already exists and is not empty", file=sys.stderr)
        exit(1)

    if not os.path.exists(args.input):
        print(f"{args.input} does not exist", file=sys.stderr)
        exit(1)

    if os.path.isdir(args.input) and len(os.listdir(args.input)) == 0:
        print(f"{args.input} is an empty directory", file=sys.stderr)
        exit(1)

    try:
        d = torch.device(int(args.device))
        args.device = f"{d.type}:{d.index}"
    except ValueError:
        pass

    torch.set_grad_enabled(False)

    if args.checkpoint is None:
        with open(hf_hub_download("IntegratedBiosciences/monet", "config.json")) as f:
            config = json.load(f)

        model = Monet(MonetConfig(**config)).eval().to(args.device)
        model.load_state_dict(torch.load(hf_hub_download("IntegratedBiosciences/monet", "model.pt"), map_location='cpu'))
    else:
        with open(args.checkpoint.replace('.pt', '_config.json')) as f:
            config = json.load(f)

        model = Monet(MonetConfig(**config)).eval().to(args.device)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    image_size = config['suggested_image_size']

    inputs = defaultdict(lambda: [None]*7)
    for file in sorted(glob.glob(os.path.join(args.input, '*'))) if os.path.isdir(args.input) else [args.input]:
        prefix, channel = re.match(r'(.+?)(_brightfield_reference|_mito|_rna|_er|_dna|_agp|_brightfield)?$', os.path.splitext(os.path.basename(file))[0]).groups()
        channel = channel or '_brightfield'
        # this defines the order of the conditioning channels given to the model
        inputs[prefix][['_brightfield_reference', '_mito', '_rna', '_er', '_dna', '_agp', '_brightfield'].index(channel)] = file

    os.makedirs(args.output, exist_ok=True)

    for file_idx in tqdm(list(range(0, len(inputs), args.batch_size))):
        batch_size_ = min(file_idx + args.batch_size, len(inputs)) - file_idx
        prefixes = []
        im = []
        mask = []

        for file_idx in range(file_idx, file_idx + batch_size_):
            prefix = list(inputs.keys())[file_idx]
            prefixes.append(prefix)
            paths = inputs[prefix]

            w, h = [Image.open(x).size for x in paths if x is not None][0]

            mask_ = [int(x is not None) for x in paths]
            im_ = [
                torch.from_numpy(np.array(Image.open(x))).to(dtype=torch.float32, device=args.device) 
                if x is not None 
                else torch.zeros((h, w), dtype=torch.float32, device=args.device)
                for x in paths
            ]
            assert all(x.shape == (h, w) for x in im_), f"All images must be grayscale and have the same shape"
            im_ = torch.stack(im_) # C, H, W

            im_ = TF.center_crop(TF.resize(im_, image_size, interpolation=TF.InterpolationMode.BILINEAR), image_size)
            im.append(im_)

            mask.append(mask_)

        mask = torch.tensor(mask, dtype=torch.long, device=args.device)
        im = torch.stack(im)

        p = [[0.02, 0.98], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.02, 0.98]]
        plow, phigh = torch.cat([
            torch.quantile(im[:, i:i+1].flatten(-2), torch.tensor(p, device=args.device), dim=-1)
            for i, p in enumerate(p)
        ], dim=-1)

        mask = torch.where(phigh > plow, mask, torch.zeros_like(mask))
        plow, phigh = plow[:, :, None, None], phigh[:, :, None, None]
        im = (im.clamp(plow, phigh) - plow) / (phigh - plow + 1e-8)

        im = im.sqrt()
        im = torch.where(mask[:, :, None, None].bool().expand_as(im), im, torch.zeros_like(im)) # zeroing out the masked channels is done _before_ the 0,1 -> -1,1
        im = (im - 0.5) * 2

        x_t = torch.randn((batch_size_, 5, image_size, image_size), device=args.device)
        timesteps = torch.linspace(1, 0, args.diffusion_steps+1, device=args.device)[:-1, None].expand(-1, batch_size_)
        encoder_hidden_states = torch.zeros((batch_size_,1,1), device=args.device)

        for t in timesteps:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_pred = model.decoder(
                    sample=torch.cat([im, x_t], dim=1),
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample[:, -5:]

            x_t = x_t - (1/args.diffusion_steps) * v_pred

        im_pred = x_t
        im_pred = (((im_pred / 2) + 0.5)**2).clamp(0, 1)
        im_rgb = torch.zeros((im_pred.shape[0], 3, im_pred.shape[2], im_pred.shape[3]), device=args.device)
        color_map = torch.tensor([
          [0, 1, 0],
          [1, 0, 0],
          [1, 1, 0],
          [0, 0, 1],
          [0, 1, 1],
        ], device=args.device)
        for channel_idx in range(im_pred.shape[1]):
            im_rgb += im_pred[:, channel_idx:channel_idx+1, :, :] * color_map[channel_idx][None, :, None, None]

        im_rgb_min, im_rgb_max = im_rgb.flatten(-2).min(dim=-1)[0][:, :, None, None], im_rgb.flatten(-2).max(dim=-1)[0][:, :, None, None]
        im_rgb = (im_rgb - im_rgb_min) / (im_rgb_max - im_rgb_min + 1e-8)
        im_rgb = (im_rgb.clamp(0,1)*255).permute(0, 2, 3, 1).to(torch.uint8).cpu()

        im_pred = (im_pred.clamp(0,1)*255).to(torch.uint8).cpu()

        for prefix, im_rgb, im_pred in zip(prefixes, im_rgb, im_pred):
            Image.fromarray(im_rgb.numpy()).save(os.path.join(args.output, f"{prefix}.png"))

            for channel, channel_name in zip(im_pred, ['mito', 'rna', 'er', 'dna', 'agp']):
                Image.fromarray(channel.numpy()).save(os.path.join(args.output, f"{prefix}_{channel_name}.png"))


if __name__ == "__main__":
    main()

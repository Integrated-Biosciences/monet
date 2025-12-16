import argparse
import glob
import json
import os
import sys

import imageio
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from monet import Monet, MonetConfig


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=r"""
Generate temporally consistent cell paint videos from brightfield time-lapse data. Uses reference conditioning to maintain visual consistency across frames.

```bash
# From a video file
python generate_cellpaint_video.py timelapse.mp4 -o cellpaint_video.mp4

# From a directory of ordered frames
python generate_cellpaint_video.py frames_folder/ -o cellpaint_video.mp4

# With custom settings
python generate_cellpaint_video.py frames_folder/ -o output.mp4 --checkpoint my_model.pt --diffusion_steps 20
```

**Input folder layout (frames in lexicographic order):**
```
frames_folder/
├── frame_0000.png     # grayscale brightfield frame 1
├── frame_0001.png     # grayscale brightfield frame 2
├── frame_0002.png     # grayscale brightfield frame 3
├── frame_0003.png     # grayscale brightfield frame 4
...
└── frame_0099.png     # grayscale brightfield frame 100
```

**Note:** All input images must be grayscale. The first frame is generated unconditionally, then subsequent frames are conditioned on the first frame generation to maintain temporal consistency.
""")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--diffusion_steps", type=int, default=10, help='Number of diffusion steps to use.')
    parser.add_argument("input", type=str, help="Single brightfield video or directory of images to be treated as a video and cellpainted. "
                                                "When providing a directory, images will be assumed to be brightfield frames in lexicographic order.")
    parser.add_argument("-o", "--output", type=str, default="output.mp4", 
                                            help="Output video. Must not exist.")
    parser.add_argument("--checkpoint", type=str, default=None, required=False, help="Optional path to local checkpoint to load. If not provided, uses the base model.")

    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"{args.output} already exists", file=sys.stderr)
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

    if os.path.isdir(args.input):
        inputs = sorted(glob.glob(os.path.join(args.input, '*')))
    else:
        inputs = imageio.mimread(args.input, memtest=False)
        inputs = [Image.fromarray(x).convert('L') for x in inputs]
        
    batch_size_ = 1
    x_t_orig = torch.randn((batch_size_, 5, image_size, image_size), device=args.device)
    reference_image = None
        
    im_pred_frames = []

    for x in tqdm(inputs):
        im = []
        mask = []

        mask.append(1)
        if isinstance(x, str):
            x = Image.open(x)
        x = torch.from_numpy(np.array(x))
        assert x.ndim == 2, f"Input must be grayscale, got {x.shape}"
        x = x.to(dtype=torch.float32, device=args.device).reshape(-1, x.shape[-2], x.shape[-1])
        x = TF.resize(x, size=image_size, interpolation=TF.InterpolationMode.BILINEAR)
        x = TF.center_crop(x, image_size)
        im.append(x)

        mask = torch.tensor(mask, dtype=torch.long, device=args.device).reshape(batch_size_, -1)
        im = torch.stack(im).reshape(batch_size_, -1, image_size, image_size)

        p = [[0.02, 0.98]]
        plow, phigh = torch.cat([
            torch.quantile(im[:, i:i+1].flatten(-2), torch.tensor(p, device=args.device), dim=-1)
            for i, p in enumerate(p)
        ], dim=-1)

        mask = torch.where(phigh > plow, mask, torch.zeros_like(mask))
        plow, phigh = plow[:, :, None, None], phigh[:, :, None, None]
        im = (im.clamp(plow, phigh) - plow) / (phigh - plow + 1e-8)

        im = im.sqrt()
        im = torch.where(mask[:, :, None, None].bool().expand_as(im), im, torch.zeros_like(im))
        im = (im - 0.5) * 2

        x_t = x_t_orig.clone()
        timesteps = torch.linspace(1, 0, args.diffusion_steps+1, device=args.device)[:-1, None].expand(-1, batch_size_)
        encoder_hidden_states = torch.zeros((batch_size_, 1, 1), device=args.device)

        # Build conditioning: reference image (6 channels) + brightfield (1 channel)
        if reference_image is not None:
            reference_image_ = reference_image
        else:
            reference_image_ = torch.full((batch_size_, 6, image_size, image_size), -1, device=args.device)

        for t in timesteps:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_pred = model.decoder(
                    sample=torch.cat([reference_image_, im, x_t], dim=1),
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample[:, -5:]

            x_t = x_t - (1/args.diffusion_steps) * v_pred

        im_pred_frames.append(x_t)

        if reference_image is None:
            reference_image = torch.cat([im, x_t], dim=1)

    n_frames = len(im_pred_frames)

    im_pred = torch.cat(im_pred_frames, dim=0)  # n_frames, 5, H, W
    im_pred = (((im_pred / 2) + 0.5)**2).clamp(0, 1)

    im_rgb = torch.zeros((n_frames, 3, image_size, image_size), device=args.device)
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
    im_rgb = (im_rgb.clamp(0, 1)*255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

    with imageio.get_writer(args.output, fps=2) as writer:
        for frame in im_rgb:
            writer.append_data(frame)

if __name__ == "__main__":
    main()

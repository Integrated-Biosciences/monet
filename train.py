import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from monet import Monet, MonetConfig


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=r"""
Fine-tune MONET on your own paired brightfield/cell paint data.

```bash
# Basic training (multi-GPU)
torchrun --nproc_per_node=4 train.py --input training_data/

# With custom settings
torchrun --nproc_per_node=4 train.py \
    --input training_data/ \
    --experiment my_finetune \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-5 \
    --train_steps 10000 \
    --save_and_eval_every_n_steps 500

# With separate evaluation data
torchrun --nproc_per_node=4 train.py \
    --input training_data/ \
    --eval_input eval_data/ \
    --experiment my_finetune
```

**Training data folder layout:**
```
training_data/
├── sample001_brightfield.png   # brightfield for sample 1
├── sample001_mito.png          # mitochondria channel for sample 1
├── sample001_rna.png           # RNA channel for sample 1
├── sample001_er.png            # ER channel for sample 1
├── sample001_dna.png           # DNA channel for sample 1
├── sample001_agp.png           # AGP/cytoskeleton channel for sample 1
├── sample002_brightfield.png   # brightfield for sample 2
├── sample002_mito.png
├── sample002_rna.png
├── sample002_er.png
├── sample002_dna.png
├── sample002_agp.png
...
```

**Evaluation data:**

If `--eval_input` is not specified, the first 4 brightfields from `--input` are automatically used for evaluation. To use custom evaluation data, specify `--eval_input` with a folder following the same format as `generate_cellpaint.py`:

```
eval_data/
├── eval001.png                        # brightfield only (unconditional generation)
├── eval002.png
├── eval003_brightfield.png            # or with reference conditioning
├── eval003_brightfield_reference.png
├── eval003_mito.png
├── eval003_rna.png
├── eval003_er.png
├── eval003_dna.png
└── eval003_agp.png
```

**Requirements:**
- All images must be grayscale and have the same dimensions within each sample
- Files must have the correct suffixes: `_brightfield`, `_mito`, `_rna`, `_er`, `_dna`, `_agp`
- Files are grouped by their prefix (everything before the suffix)
- Missing channels are allowed and will be passed as blank images

**Output structure:**
```
my_finetune/
├── step_0001000.pt                    # model checkpoint
├── step_0001000_config.json           # model config
├── step_0001000_optim.pt              # optimizer state
├── step_0001000_eval_input/           # evaluation inputs (if not specified)
├── step_0001000_eval_output/          # generated evaluation outputs
├── step_0002000.pt
...
```

Training progress is logged to Weights & Biases under the project `monet_finetuning`.
    """)
    # target a global batch size of ~16
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_and_eval_every_n_steps", type=int, default=1000)
    parser.add_argument("--experiment", type=str, required=False, default=None, help="Output directory and wandb run name. If not provided, will use a timestamp.")
    parser.add_argument("--input", type=str, help="Directory of brightfields and cellpaints. Files must specify which image they are a part of with their prefix " 
                                                "and which channel they are with their suffix. Expected suffixes are _brightfield, _mito, _rna, _er, _dna, and _agp -- all being their corresponding channels of "
                                                "a cellpaint and corresponding brightfield. "
                                                "i.e. <input>/00000_brightfield.png <input>/00000_mito.png <input>/00000_rna.png <input>/00000_er.png <input>/00000_dna.png <input>/00000_agp.png is a "
                                                "single image with all channels specified. Any not specified channels will be passed as blank images.")
    parser.add_argument("--eval_input", type=str, required=False, default=None, help="Passed to `generate_cellpaint.py`. If not provided, will evaluate on the first 4 brightfields in `--input`. "
                                                "Single brightfield image or directory of channels to be cellpainted. When providing a directory, files must specify which image they are a part of with their prefix " 
                                                "and which channel they are with their suffix. Channels with no suffix specified will be treated as brightfield channels. "
                                                "Any not specified channels will not be used for conditioning. Expected suffixes are _brightfield_reference, _mito, _rna, _er, _dna, _agp, and _brightfield. "
                                                "_brightfield suffix is the brightfield to cellpaint. _brightfield_reference suffix is the brightfield of the reference conditioning image. "
                                                "i.e. <input>/00000.png <input>/00001.png <input>/00002.png <input>/00003.png is four separate brightfields to be cellpainted. "
                                                "<input>/00000_brightfield_reference.png <input>/00000_mito.png <input>/00000_rna.png <input>/00000_er.png <input>/00000_dna.png <input>/00000_agp.png <input>/00000_brightfield.png is a "
                                                "single image to be cellpainted with a specified reference image to condition the generation on."
                        )
    parser.add_argument("--train_steps", type=int, required=False, default=None, help="Number of steps to train for. If not provided, will train indefinitely.")
                                                
    args = parser.parse_args()

    if not (os.path.isdir(args.input) and len(os.listdir(args.input)) > 0):
        print(f"{args.input} must be a non-empty directory", file=sys.stderr)
        exit(1)

    if args.experiment is None:
        args.experiment = f"train_{time.strftime('%Y%m%d_%H%M%S')}"

    if not ("RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_WORLD_SIZE" in os.environ):
        print("must run script with torchrun --nproc_per_node=<num_gpus> train.py ...", file=sys.stderr)
        exit(1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
    device = 'cuda'

    with open(hf_hub_download("IntegratedBiosciences/monet", "config.json")) as f:
        config = json.load(f)

    image_size = config['suggested_image_size']

    model = Monet(MonetConfig(**config)).eval().to(device)
    model.load_state_dict(torch.load(hf_hub_download("IntegratedBiosciences/monet", "model.pt"), map_location='cpu'))

    model.decoder = DDP(model.decoder, device_ids=[local_rank], find_unused_parameters=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    inputs = defaultdict(lambda: [None]*6)
    for file in sorted(glob.glob(os.path.join(args.input, '*'))) if os.path.isdir(args.input) else [args.input]:
        match = re.match(r'(.+?)(_brightfield|_mito|_rna|_er|_dna|_agp)$', os.path.splitext(os.path.basename(file))[0])
        if match is None:
            print(f"Could not parse <suffix>_<brightfiled|mito|rna|er|dna|agp> out of {file}", file=sys.stderr)
            exit(1)
        prefix, channel = match.groups()
        inputs[prefix][['_brightfield', '_mito', '_rna', '_er', '_dna', '_agp'].index(channel)] = file

    if local_rank == 0:
        print(f"Found {len(inputs)} datapoints in {args.input}")

    current_epoch = []

    if rank == 0:
        wandb.init(project="monet_finetuning", name=args.experiment, config=vars(args))

    step = 1

    while True:
        t0 = time.perf_counter()
        total_loss = 0.0

        for _ in range(args.grad_accum_steps):
            with torch.no_grad():
                batch_size_ = 0
                im = []
                mask = []

                while batch_size_ < args.batch_size:
                    if len(current_epoch) == 0:
                        current_epoch = list(inputs.keys())
                        random.shuffle(current_epoch)
                    paths = inputs[current_epoch.pop(0)]

                    w, h = [Image.open(x).size for x in paths if x is not None][0]

                    mask_ = [int(x is not None) for x in paths]
                    im_ = [
                        torch.from_numpy(np.array(Image.open(x))).to(dtype=torch.float32, device=device) 
                        if x is not None 
                        else torch.zeros((h, w), dtype=torch.float32, device=device)
                        for x in paths
                    ]
                    assert all(x.shape == (h, w) for x in im_), f"All images must be grayscale and have the same shape"
                    im_ = torch.stack(im_) # C, H, W

                    reference_image = augment_image(im_, image_size)
                    if random.random() < 0.8:
                        target_image = augment_image(im_, image_size)
                    else:
                        target_image = TF.center_crop(TF.resize(im_, image_size, interpolation=TF.InterpolationMode.BILINEAR), image_size)

                    im_ = torch.cat([reference_image, target_image], dim=0)
                    im.append(im_)

                    mask.append(mask_ + mask_)

                    batch_size_ += 1

                mask = torch.tensor(mask, dtype=torch.long, device=device)
                im = torch.stack(im)

                p = [[0.02, 0.98], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99],
                     [0.02, 0.98], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99]]
                plow, phigh = torch.cat([
                    torch.quantile(im[:, i:i+1].flatten(-2), torch.tensor(p, device=device), dim=-1)
                    for i, p in enumerate(p)
                ], dim=-1)

                mask = torch.where(phigh > plow, mask, torch.zeros_like(mask))
                plow, phigh = plow[:, :, None, None], phigh[:, :, None, None]
                im = (im.clamp(plow, phigh) - plow) / (phigh - plow + 1e-8)

                im = im.sqrt()
                im = torch.where(mask[:, :, None, None].bool().expand_as(im), im, torch.zeros_like(im)) # zeroing out the masked channels is done _before_ the 0,1 -> -1,1
                im = (im - 0.5) * 2

                reference_image = im[:, :6]
                brightfield_conditioning = im[:, 6:7]
                x0 = im[:, 7:]
                noise = torch.randn_like(x0)
                v_target = noise - x0

                timesteps = torch.rand(batch_size_, device=device)
                timesteps[torch.rand(batch_size_, device=device) < 0.02] = 1.0 # Set 2% of timesteps to 1.0
                x_t = noise * timesteps[:, None, None, None] + x0 * (1 - timesteps[:, None, None, None])

                encoder_hidden_states = torch.zeros((batch_size_,1,1), device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_pred = model.decoder(
                    sample=torch.cat([
                        reference_image if random.random() < 0.95 else torch.full_like(reference_image, -1.0), 
                        brightfield_conditioning if random.random() < 0.95 else torch.full_like(brightfield_conditioning, -1.0), 
                        x_t
                    ], dim=1),
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample[:, -5:].float()
            loss = F.mse_loss(v_pred[mask[:, -5:].bool()], v_target[mask[:, -5:].bool()]) / args.grad_accum_steps
            loss.backward()
            total_loss += loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(grad_norm, op=dist.ReduceOp.AVG)

        dist.barrier()

        step_time = time.perf_counter() - t0
        steps_per_hour = 3600 / step_time

        if rank == 0:
            wandb.log({"loss": total_loss.item(), "grad_norm": grad_norm.item(), "step_time": step_time, "steps_per_hour": steps_per_hour}, step=step)

        if local_rank == 0:
            print(f"step {step:07d} | loss {total_loss.item():07.4f} | grad_norm {grad_norm.item():07.4f} | step_time {step_time:02.2f} | steps_per_hour {steps_per_hour:06.2f}")

        if args.save_and_eval_every_n_steps is not None and step % args.save_and_eval_every_n_steps == 0 and rank == 0:
            os.makedirs(args.experiment, exist_ok=True)

            model_sd = {f'decoder.{k}': v.cpu() for k, v in model.decoder.module.state_dict().items()}

            optim_sd = optim.state_dict()
            optim_sd['state'] = {k: {k2: v2.cpu() for k2, v2 in v.items()} for k, v in optim_sd['state'].items()}

            torch.save(model_sd, os.path.join(args.experiment, f"step_{step:07d}.pt"))
            torch.save(optim_sd, os.path.join(args.experiment, f"step_{step:07d}_optim.pt"))
            with open(os.path.join(args.experiment, f"step_{step:07d}_config.json"), "w") as f:
                json.dump(config, f)

            if args.eval_input is None:
                eval_input = os.path.join(args.experiment, f"step_{step:07d}_eval_input")
                os.makedirs(eval_input)
                eval_brightfields = [x[0] for x in list(inputs.values()) if x[0] is not None][:4]
                assert len(eval_brightfields), "Could not find any brightfields for evaluation"
                for x in eval_brightfields:
                    shutil.copy(x, eval_input)
            else:
                eval_input = args.eval_input

            eval_output = os.path.join(args.experiment, f"step_{step:07d}_eval_output")

            assert os.system(f"python generate_cellpaint.py -o {eval_output} --checkpoint {os.path.join(args.experiment, f'step_{step:07d}.pt')} --device {local_rank} {eval_input}") == 0

            eval_input_paths = defaultdict(lambda: [None]*7)
            for file in sorted(glob.glob(os.path.join(eval_input, '*'))) if os.path.isdir(eval_input) else [eval_input]:
                prefix, channel = re.match(r'(.+?)(_brightfield_reference|_mito|_rna|_er|_dna|_agp|_brightfield)?$', os.path.splitext(os.path.basename(file))[0]).groups()
                channel = channel or '_brightfield'
                eval_input_paths[prefix][['_brightfield_reference', '_mito', '_rna', '_er', '_dna', '_agp', '_brightfield'].index(channel)] = file

            for i, prefix in enumerate(eval_input_paths.keys()):
                generated = np.array(Image.open(os.path.join(eval_output, f"{prefix}.png")))
                h, w = generated.shape[:2]

                brightfield = Image.open(eval_input_paths[prefix][-1])
                brightfield = brightfield.resize((w,h))
                brightfield = np.array(brightfield)
                assert brightfield.ndim == 2
                # 8 bit rgb conversion
                brightfield = (brightfield.astype(np.float32) / np.iinfo(brightfield.dtype).max) * 255
                brightfield = brightfield.astype(np.uint8)
                brightfield = np.repeat(brightfield[:, :, None], 3, axis=2)

                combined = Image.fromarray(np.concatenate([brightfield, generated], axis=1))
                wandb.log({f"eval_{i}": wandb.Image(combined)}, step=step)

        dist.barrier()
        step += 1

        if args.train_steps is not None and step > args.train_steps:
            break

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()

def augment_image(image, target_size):
    import random

    import torchvision.transforms.functional as TF

    assert image.ndim == 3, image.shape

    # image: Tensor [C,H,W], float
    _, h, w = image.shape

    # flips
    if random.random() < 0.2:
        image = TF.hflip(image) if random.random() < 0.5 else TF.vflip(image)

    # zoom
    _, h, w = image.shape  # refresh dims
    if random.random() < 0.1:
        if random.random() < 0.5:
            scale_factor = random.uniform(1.0, 1.33)
        else:
            min_scale = target_size / min(h, w)
            max_scale = max(min_scale, 0.67)
            scale_factor = random.uniform(max_scale, 1.0)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        image = TF.resize(image, (new_h, new_w), antialias=True)

    # ensure at least target size
    _, h, w = image.shape
    if h < target_size or w < target_size:
        scale = max(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = TF.resize(image, (new_h, new_w), antialias=True)
        _, h, w = image.shape

    # random crop
    top = random.randint(0, h - target_size)
    left = random.randint(0, w - target_size)
    image = TF.crop(image, top, left, target_size, target_size)

    return image

if __name__ == "__main__":
    main()
# Monet

MONET (Morphological Observation Neural Enhancement Tool) is a diffusion model that generates virtual cell paint images from brightfield microscopy. Cell painting is a popular technique for creating high-contrast images of cell morphology, but it is labor-intensive and requires chemical fixation—making time-lapse studies impossible. MONET bypasses these limitations by predicting five cell paint channels (DNA, RNA, ER, cytoskeleton/AGP, and mitochondria) directly from brightfield images.

The model uses a reference consistency architecture that enables artifact-free generation of time-lapse videos, despite the fact that paired (brightfield, cell paint) video training data cannot exist. At inference time, the first frame is generated unconditionally, then subsequent frames are conditioned on that first frame to maintain visual consistency. This architecture also enables a form of in-context learning for domain adaptation to new cell lines and imaging hardware.
MONET is a 350M parameter UNet-based diffusion model trained on 8M+ images from the Broad Cell Paint Gallery. Pre-trained weights are available on HuggingFace at [IntegratedBiosciences/monet](https://huggingface.co/IntegratedBiosciences/monet).

📄 Paper: [arXiv:2512.11928](https://arxiv.org/abs/2512.11928)
🖼️ Examples: [thiscellpaintingdoesnotexist.com](https://thiscellpaintingdoesnotexist.com)
🤗 Model: [IntegratedBiosciences/monet](https://huggingface.co/IntegratedBiosciences/monet)

## Example environment setup with uv venv
```bash
uv venv --python 3.12 monet_venv
source monet_venv/bin/activate
uv pip install --torch-backend=auto -r requirements.txt
```

## Example environment setup with docker
```bash
docker build -t monet .
docker run -it --rm --gpus all -v $(pwd):/workspace monet
```

## generate_cellpaint.py

Generate virtual cell paint images from brightfield microscopy images.

```bash
# Single brightfield image (simplest usage)
python generate_cellpaint.py my_brightfield.png -o output/

# Directory of multiple brightfield images
python generate_cellpaint.py input_folder/ -o output/

# With custom checkpoint and settings
python generate_cellpaint.py input_folder/ -o output/ --checkpoint my_model.pt --diffusion_steps 20 --batch_size 4
```

**Input folder layouts:**

*Multiple brightfield images (no suffixes = treated as brightfield)*
```
input_folder/
├── 00000.png          # brightfield image 1
├── 00001.png          # brightfield image 2
├── 00002.png          # brightfield image 3
└── 00003.png          # brightfield image 4
```

*With reference conditioning (for consistent style/domain adaptation)*
```
input_folder/
├── sample1_brightfield.png            # brightfield to cellpaint
├── sample1_brightfield_reference.png  # reference brightfield for conditioning
├── sample1_mito.png                   # reference mitochondria channel
├── sample1_rna.png                    # reference RNA channel
├── sample1_er.png                     # reference ER channel
├── sample1_dna.png                    # reference DNA channel
└── sample1_agp.png                    # reference AGP/cytoskeleton channel
```

**Example input/output:**

*Input:*
```
input_folder/
├── 00000.png          # brightfield image with prefix "00000"
├── 00001.png          # brightfield image with prefix "00001"
└── 00002.png          # brightfield image with prefix "00002"
```

*Output:*
```
output/
├── 00000.png          # composite RGB image
├── 00000_mito.png     # mitochondria channel
├── 00000_rna.png      # RNA channel
├── 00000_er.png       # ER channel
├── 00000_dna.png      # DNA channel
├── 00000_agp.png      # AGP/cytoskeleton channel
├── 00001.png
├── 00001_mito.png
├── 00001_rna.png
├── 00001_er.png
├── 00001_dna.png
├── 00001_agp.png
├── 00002.png
├── 00002_mito.png
...
```

---

## generate_cellpaint_video.py

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

---

## train.py

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

# Licensing
The model weights at [IntegratedBiosciences/monet](https://huggingface.co/IntegratedBiosciences/monet) are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

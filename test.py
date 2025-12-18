"""
This is a set of smoke tests that run the scripts with some example data.
"""

import os

if not os.path.exists("monet_example_data_cpg0002/"):
    assert os.system("wget https://github.com/Integrated-Biosciences/monet/releases/download/test_data/monet_example_data_cpg0002.tar") == 0
    assert os.system("tar -xvf monet_example_data_cpg0002.tar") == 0

assert os.system(r"""torchrun --nproc_per_node=1 train.py \
                 --input monet_example_data_cpg0002/example_training_data \
                 --batch_size 2 --grad_accum_steps 1 --experiment train_1 --save_and_eval_every_n_steps 2 \
                 --train_steps 3
                  """) == 0
assert os.system("rm -rf train_1") == 0

assert os.system(r"""torchrun --nproc_per_node=1 train.py \
                 --input monet_example_data_cpg0002/example_training_data_8bit \
                 --batch_size 2 --grad_accum_steps 1 --experiment train_2 --save_and_eval_every_n_steps 2 \
                 --train_steps 3
                  """) == 0
assert os.system("rm -rf train_2") == 0

assert os.system(r"""python generate_cellpaint.py \
                 -o output_1 \
                 monet_example_data_cpg0002/example_generate_data_brightfield_only
                 """) == 0
assert os.system("rm -rf output_1") == 0

assert os.system(r"""python generate_cellpaint.py \
                 -o output_2 \
                 monet_example_data_cpg0002/example_generate_data_brightfield_only_8bit
                 """) == 0
assert os.system("rm -rf output_2") == 0

assert os.system(r"""python generate_cellpaint.py \
                 -o output_3 \
                 monet_example_data_cpg0002/example_generate_data_brightfield_no_suffix
                 """) == 0
assert os.system("rm -rf output_3") == 0

assert os.system(r"""python generate_cellpaint.py \
                 -o output_4 \
                 monet_example_data_cpg0002/example_generate_data_brightfield_no_suffix_8bit
                 """) == 0
assert os.system("rm -rf output_4") == 0

assert os.system(r"""python generate_cellpaint.py \
                 -o output_5 \
                 monet_example_data_cpg0002/example_generate_data_reference_image
                 """) == 0
assert os.system("rm -rf output_5") == 0

assert os.system(r"""python generate_cellpaint.py \
                 -o output_6 \
                 monet_example_data_cpg0002/example_generate_data_reference_image_8bit
                 """) == 0
assert os.system("rm -rf output_6") == 0

assert os.system(r"""python generate_cellpaint_video.py \
                 --output output_video1.mp4 \
                monet_example_data_cpg0002/video_input_frames
                 """) == 0
assert os.system("rm -rf output_video1.mp4") == 0

assert os.system(r"""python generate_cellpaint_video.py \
                 --output output_video2.mp4 \
                 monet_example_data_cpg0002/video_input.mp4
                 """) == 0
assert os.system("rm -rf output_video2.mp4") == 0

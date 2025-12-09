import torch
import torch.nn as nn
from tqdm import tqdm

class MonetConfig:
    def __init__(
        self,
        num_channels,
        block_out_channels=[128, 256, 512],
        down_blocks = [
            'DownBlock2D',
            'DownBlock2D',
            'DownBlock2D',
        ],
        up_blocks = [
            'UpBlock2D',
            'UpBlock2D',
            'UpBlock2D'
        ],
        layers_per_block=4,
        transformer_layers_per_block=4,
        suggested_image_size=512,
    ):
        self.num_channels = num_channels
        self.block_out_channels = block_out_channels
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.layers_per_block = layers_per_block
        self.transformer_layers_per_block = transformer_layers_per_block
        self.suggested_image_size = suggested_image_size

class Monet(nn.Module):
    def __init__(self, config):
        super().__init__()
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        self.config = config
        self.num_channels = config.num_channels

        self.decoder = UNet2DConditionModel(
            **{
                'sample_size': 64, # I think is supposed to mean the image size but is only used in pipeline class, not model
                'in_channels': self.num_channels * 2, 
                'out_channels': self.num_channels * 2,  
                'center_input_sample': False,
                'flip_sin_to_cos': True,
                'freq_shift': 0,
                'down_block_types': config.down_blocks,
                'mid_block_type': 'UNetMidBlock2DCrossAttn',
                'up_block_types': config.up_blocks,
                'only_cross_attention': False,
                'block_out_channels': config.block_out_channels,
                'layers_per_block': config.layers_per_block,
                'downsample_padding': 1,
                'mid_block_scale_factor': 1,
                'dropout': 0.0,
                'act_fn': 'silu',
                'norm_num_groups': 32,
                'norm_eps': 1e-05,
                'cross_attention_dim': 1,  # Not using cross attention
                'transformer_layers_per_block': config.transformer_layers_per_block,
                'reverse_transformer_layers_per_block': None,
                'encoder_hid_dim': None,
                'encoder_hid_dim_type': None,
                'attention_head_dim': 8,
                'num_attention_heads': None,
                'dual_cross_attention': False,
                'use_linear_projection': False,
                'class_embed_type': None,
                'addition_embed_type': None,
                'addition_time_embed_dim': None,
                'num_class_embeds': None,
                'upcast_attention': False,
                'resnet_time_scale_shift': 'default',
                'resnet_skip_time_act': False,
                'resnet_out_scale_factor': 1.0,
                'time_embedding_type': 'positional',
                'time_embedding_dim': None,
                'time_embedding_act_fn': None,
                'timestep_post_act': None,
                'time_cond_proj_dim': None,
                'conv_in_kernel': 3,
                'conv_out_kernel': 3,
                'projection_class_embeddings_input_dim': None,
                'class_embeddings_concat': False,
                'mid_block_only_cross_attention': None,
                'cross_attention_norm': None,
                'addition_embed_type_num_heads': 64,
                'attention_type': 'flash'
            }
        )

        total_params = sum(p.numel() for p in self.decoder.parameters())
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Total Params (Millions): {(total_params/1e6):.2f}")

    def forward(self,
        images_to_predict, 
        images_to_predict_channel_mask, # (BS, num_channels) bool, True means image has the channel
    ):
        # make sure brightfield mask is 0
        # imasges come in as [0,1], set images to [-1,1]
        images_to_predict = (images_to_predict - .5) * 2

        # Generate the Flow matching objective
        bs = images_to_predict.shape[0]
        noise = torch.randn_like(images_to_predict)
        # Sample timesteps: 2% chance of timestep=1, otherwise random [0,1)
        timesteps = torch.rand(bs, device=images_to_predict.device)
        # Set 2% of timesteps to 1.0
        max_timestep_mask = torch.rand(bs, device=images_to_predict.device) < 0.02
        timesteps[max_timestep_mask] = 1.0
        
        # Don't noise the brighfield
        noised_images = images_to_predict * (1-timesteps[:, None, None, None]) + noise * timesteps[:, None, None, None]

        # Set noised_images to 0 for channels where mask is False
        mask_expanded = images_to_predict_channel_mask.unsqueeze(-1).unsqueeze(-1)  # (BS, C, 1, 1)
        noised_images = noised_images * mask_expanded  # Zero out masked channels
        
        # Channels should be {brighfield reference, cellpaint reference, brightfield target, cellpaint target}
        # we only want to noise and denoise the cellpaint target
        reference_channels = self.num_channels + 1
        noised_images[:,0:reference_channels,:,:] = images_to_predict[:,0:reference_channels,:,:].clone().detach()

        v_target = noise - images_to_predict

        # Create dummy encoder hidden states (required by UNet2DConditionModel)
        # Shape: (batch, seq_len, cross_attention_dim) where cross_attention_dim=1
        batch_size = noised_images.shape[0]
        dummy_encoder_hidden_states = torch.zeros(
            batch_size, 1, 1,  # (batch, seq_len=1, cross_attention_dim=1)
            device=noised_images.device,
            dtype=noised_images.dtype
        )

        decoder_out = self.decoder(
            sample=noised_images,
            timestep=timesteps,
            encoder_hidden_states=dummy_encoder_hidden_states,
        )
        
        # Take just the second half
        v_pred = decoder_out.sample
    
        delta = (v_pred - v_target)
        # Expand channel mask to match image dimensions: (BS, C) -> (BS, C, H, W)
        images_to_predict_channel_mask[:,0:reference_channels] = False # No loss over reference channels
        expanded_mask = images_to_predict_channel_mask.unsqueeze(-1).unsqueeze(-1)  # (BS, C, 1, 1)
        expanded_mask = expanded_mask.expand_as(delta)  # (BS, C, H, W)

        # Calculate MSE only over valid (masked) pixels
        # This gives mean squared error per valid pixel
        masked_delta_squared = (delta ** 2) * expanded_mask
        flow_mse = masked_delta_squared.sum() / expanded_mask.sum()

        return {
            'flow_mse': flow_mse,
            'v_pred': v_pred,
        }

    def generate(self, reference_image, brightfield_channel, sample_steps=50, cfg=1.0):
        bs, h, w = brightfield_channel.shape
        print(f"Brightfield shape: {brightfield_channel.shape}")
        brightfield_channel = (brightfield_channel - .5) * 2
        reference_image = (reference_image - .5) * 2
        num_channels = self.num_channels

        # Initialize with random noise for the channels to generate
        generated_channels = torch.randn(
            bs, self.num_channels * 2, h, w,
            device=brightfield_channel.device,
            dtype=brightfield_channel.dtype
        )
        generated_channels[:,0:self.num_channels,:,:] = reference_image.clone()
        generated_channels[:,self.num_channels,:,:] = brightfield_channel.clone()

        for i in tqdm(list(range(sample_steps, 0, -1))):
            t = torch.full((bs,), i / sample_steps, device=brightfield_channel.device)

            # Stack reference and current generated channels before each step
            # This is crucial because generated_channels gets overwritten each iteration
            # Create dummy encoder hidden states
            dummy_encoder_hidden_states = torch.zeros(
                bs, 1, 1,
                device=brightfield_channel.device,
                dtype=brightfield_channel.dtype
            )

            with torch.autocast('cuda', dtype=torch.bfloat16):
                decoder_out = self.decoder(
                    sample=generated_channels,
                    timestep=t,
                    encoder_hidden_states=dummy_encoder_hidden_states,
                )

            # Extract velocity prediction for generated channels only (second half)
            v_pred = decoder_out.sample  # Take second half of channels

            # Update generated channels using flow matching step
            generated_channels = generated_channels - (1/sample_steps) * v_pred
            generated_channels[:,0:num_channels,:,:] = reference_image.clone()
            generated_channels[:,num_channels,:,:] = brightfield_channel.clone()

        print(f"Generation min and max values: {generated_channels.min()} | {generated_channels.max()}")
        # this is is in -1, 1, convert to [0,1] and square to get back the original distribution
        generated_channels = ((generated_channels.clip(-1, 1) / 2 + .5).squeeze()) ** 2
        print(f"After back to [0,1: min {generated_channels.min()} | max {generated_channels.max()}")

        generated_channels_dict = {
            'mito': generated_channels[num_channels+1,:,:].float().cpu().data.numpy(),
            'rna': generated_channels[num_channels+2,:,:].float().cpu().data.numpy(),
            'er': generated_channels[num_channels+3,:,:].float().cpu().data.numpy(),
            'dna': generated_channels[num_channels+4,:,:].float().cpu().data.numpy(),
            'agp': generated_channels[num_channels+5,:,:].float().cpu().data.numpy(),
        }
        return generated_channels_dict

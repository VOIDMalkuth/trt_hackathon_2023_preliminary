import gradio as gr
import numpy as np
import torch
import random

from cldm.model import create_model, load_state_dict

import warnings
warnings.filterwarnings('ignore')

IMAGE_HINT_SHAPE=(1, 3, 256, 384)
X_NOISY_SHAPE=(1, 4, 32, 48)
TOKENS_SHAPE=(1, 77)
CONTEXT_SHAPE=(1, 77, 768)
TIMESTEPS_SHAPE=(1,)
CONTROL_FEATURE_SHAPES = [
    (1, 320, 32, 48), (1, 320, 32, 48), (1, 320, 32, 48),
    (1, 320, 16, 24), (1, 640, 16, 24), (1, 640, 16, 24),
    (1, 640, 8, 12), (1, 1280, 8, 12), (1, 1280, 8, 12),
    (1, 1280, 4, 6), (1, 1280, 4, 6), (1, 1280, 4, 6),
    (1, 1280, 4, 6),
]
VAE_Z_SHAPE=(1, 4, 32, 48)

def export_controlnet_onnx(control_ldm_model):
    controlnet_model = control_ldm_model.control_model
    controlnet_model.eval()
    
    hint = torch.randn(*IMAGE_HINT_SHAPE, dtype=torch.float32)
    x_noisy = torch.randn(*X_NOISY_SHAPE, dtype=torch.float32)
    timesteps = torch.tensor([500], dtype=torch.int32)
    context = torch.randn(*CONTEXT_SHAPE, dtype=torch.float32)

    outs = controlnet_model(x=x_noisy, hint=hint, timesteps=timesteps, context=context)

    # print(type(outs))
    # print(len(outs))

    torch.onnx.export(
        controlnet_model,
        (x_noisy, hint, timesteps, context),
        "onnx_models/controlnet/controlnet_static_shape.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['x_noisy', 'hint', 'timesteps', 'context'],
        output_names=[
            'control_1', 'control_2', 'control_3', 'control_4',
            'control_5', 'control_6', 'control_7', 'control_8',
            'control_9', 'control_10', 'control_11', 'control_12',
            'control_13'
        ],
        dynamic_axes={
            # inputs
            'x_noisy': {0: 'batch_size'},
            'hint': {0: 'batch_size'},
            'timesteps': {0: 'batch_size'},
            'context': {0: 'batch_size'},
            # outputs
            'control_1': {0: 'batch_size'},
            'control_2': {0: 'batch_size'},
            'control_3': {0: 'batch_size'},
            'control_4': {0: 'batch_size'},
            'control_5': {0: 'batch_size'},
            'control_6': {0: 'batch_size'},
            'control_7': {0: 'batch_size'},
            'control_8': {0: 'batch_size'},
            'control_9': {0: 'batch_size'},
            'control_10': {0: 'batch_size'},
            'control_11': {0: 'batch_size'},
            'control_12': {0: 'batch_size'},
            'control_13': {0: 'batch_size'},
        }
    )

def export_unet_onnx(control_ldm_model):
    unet_model = control_ldm_model.model.diffusion_model
    
    x_noisy = torch.randn(*X_NOISY_SHAPE, dtype=torch.float32)
    timesteps = torch.tensor([500], dtype=torch.int32)
    context = torch.randn(*CONTEXT_SHAPE, dtype=torch.float32)
    control_orig = [torch.randn(*i, dtype=torch.float32) for i in CONTROL_FEATURE_SHAPES]
    control = control_orig.copy()

    outs = unet_model(x=x_noisy, timesteps=timesteps, context=context, control=control, only_mid_control=False)

    # print(type(outs))
    # print(outs.shape)

    torch.onnx.export(
        unet_model,
        (x_noisy, timesteps, context, control_orig),
        "onnx_models/unet/unet_static_shape.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            'x_noisy', 'timesteps', 'context',
            'control_1', 'control_2', 'control_3', 'control_4',
            'control_5', 'control_6', 'control_7', 'control_8',
            'control_9', 'control_10', 'control_11', 'control_12',
            'control_13'
        ],
        output_names=[
            'eps'
        ],
        dynamic_axes={
            # inputs
            'x_noisy': {0: 'batch_size'},
            'timesteps': {0: 'batch_size'},
            'context': {0: 'batch_size'},
            'control_1': {0: 'batch_size'},
            'control_2': {0: 'batch_size'},
            'control_3': {0: 'batch_size'},
            'control_4': {0: 'batch_size'},
            'control_5': {0: 'batch_size'},
            'control_6': {0: 'batch_size'},
            'control_7': {0: 'batch_size'},
            'control_8': {0: 'batch_size'},
            'control_9': {0: 'batch_size'},
            'control_10': {0: 'batch_size'},
            'control_11': {0: 'batch_size'},
            'control_12': {0: 'batch_size'},
            'control_13': {0: 'batch_size'},
            # outputs
            'eps': {0: 'batch_size'},
        },
    )

def export_vae_onnx(control_ldm_model):
    vae_model = control_ldm_model.first_stage_model
    vae_model.eval()
    vae_model.forward = vae_model.decode

    z = torch.randn(*VAE_Z_SHAPE, dtype=torch.float32)
    outs = vae_model(z)

    torch.onnx.export(
        vae_model,
        (z),
        "onnx_models/vae/vae_static_shape.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            'z'
        ],
        output_names=[
            'x_samples'
        ],
        # verbose=True
    )
    
def export_clip_onnx(control_ldm_model):
    clip_model = control_ldm_model.cond_stage_model.transformer
    clip_model.eval()

    tokens = torch.ones(*TOKENS_SHAPE, dtype=torch.int32)
    outs = clip_model(tokens)

    torch.onnx.export(
        clip_model,
        (tokens),
        "onnx_models/clip/clip_static_shape.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            'tokens'
        ],
        output_names=[
            'cond_out', 'other_states'
        ],
        dynamic_axes={
            # inputs
            'tokens': {0: 'batch_size'},
            # outputs
            'cond_out': {0: 'batch_size'},
            'other_states': {0: 'batch_size'},
        },
    )

def main():
    control_ldm_model = create_model('./models/cldm_v15.yaml').cpu()
    control_ldm_model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
    export_controlnet_onnx(control_ldm_model)
    export_unet_onnx(control_ldm_model)
    export_vae_onnx(control_ldm_model)
    export_clip_onnx(control_ldm_model)

if __name__ == "__main__":
    main()
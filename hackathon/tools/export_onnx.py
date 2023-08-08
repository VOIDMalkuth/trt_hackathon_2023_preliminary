import gradio as gr
import numpy as np
import torch
import random

from cldm.model import create_model, load_state_dict

IMAGE_HINT_SHAPE=(1, 3, 256, 384)
X_NOISY_SHAPE=(1, 4, 32, 48)
CONTEXT_SHAPE=(1, 77, 768)
TIMESTEPS_SHAPE=(1,)
CONTROL_FEATURE_SHAPES = [
    (1, 320, 32, 48), (1, 320, 32, 48), (1, 320, 32, 48),
    (1, 320, 16, 24), (1, 640, 16, 24), (1, 640, 16, 24),
    (1, 640, 8, 12), (1, 1280, 8, 12), (1, 1280, 8, 12),
    (1, 1280, 4, 6), (1, 1280, 4, 6), (1, 1280, 4, 6),
    (1, 1280, 4, 6),
]

def export_controlnet_onnx(control_ldm_model):
    controlnet_model = control_ldm_model.control_model
    controlnet_model.eval()
    
    hint = torch.randn(*IMAGE_HINT_SHAPE, dtype=torch.float32)
    x_noisy = torch.randn(*X_NOISY_SHAPE, dtype=torch.float32)
    timesteps = torch.tensor([500], dtype=torch.int64)
    context = torch.randn(*CONTEXT_SHAPE, dtype=torch.float32)

    outs = controlnet_model(x=x_noisy, hint=hint, timesteps=timesteps, context=context)

    print(type(outs))
    print(len(outs))

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
        ]
        # dynamic_axes={
        #     'context': {1: 'seqlen'},
        # }
    )

def export_unet_onnx(control_ldm_model):
    unet_model = control_ldm_model.model.diffusion_model
    
    x_noisy = torch.randn(*X_NOISY_SHAPE, dtype=torch.float32)
    timesteps = torch.tensor([500], dtype=torch.int64)
    context = torch.randn(*CONTEXT_SHAPE, dtype=torch.float32)
    control_orig = [torch.randn(*i, dtype=torch.float32) for i in CONTROL_FEATURE_SHAPES]
    control = control_orig.copy()

    # need to change model to enable_grad
    outs = unet_model(x=x_noisy, timesteps=timesteps, context=context, control=control, only_mid_control=False)

    print(type(outs))
    print(outs.shape)

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
        ]
        # dynamic_axes={
        #     # inputs
        #     'context': {1: 'seqlen'},
        # },
        # verbose=True
    )

def main():
    control_ldm_model = create_model('./models/cldm_v15.yaml').cpu()
    control_ldm_model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    export_controlnet_onnx(control_ldm_model)
    export_unet_onnx(control_ldm_model)

if __name__ == "__main__":
    main()
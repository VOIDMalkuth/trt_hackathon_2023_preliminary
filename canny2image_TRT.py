from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

import ctypes
import cuda
from cuda import cudart

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from hackathon.trt_drivers import VaeTRT, ClipTRT
from hackathon.fused_apply_model import FusedControlnetAndUnetTrt

class hackathon():

    def initialize(self, export_calib_data=False):
        _, self.stream = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
        self.torch_stream = torch.cuda.ExternalStream(int(self.stream))
        self.torch_stream.query()
        torch.cuda.set_stream(self.torch_stream)

        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.bs = 1

        if export_calib_data:
            self.model.export_calib_data = True
        else:
            self.bs = 2
            use_cuda_graph = True
            fused_controlnet_and_unet_trt = FusedControlnetAndUnetTrt("trt_controlnet.plan", "trt_unet.plan", self.stream, bs=self.bs, use_cuda_graph=use_cuda_graph)
            vae_trt = VaeTRT("trt_vae_batch_1.plan", self.stream, bs=1, use_cuda_graph=use_cuda_graph)
            clip_trt = ClipTRT("trt_clip.plan", self.stream, bs=self.bs, use_cuda_graph=use_cuda_graph)
            self.model.updateTrtEngines({
                "FusedControlnetAndUnetTrt": fused_controlnet_and_unet_trt,
                "VAE": vae_trt,
                "CLIP": clip_trt,
                "batch_size": self.bs
            })

        self.ddim_sampler = DDIMSampler(self.model)


    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        assert eta == 0
        assert guess_mode is False
        
        with torch.no_grad():
            ddim_steps = int(ddim_steps * 0.4)
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)


            cond_prompt = [prompt + ', ' + a_prompt] * num_samples
            uncond_prompt = [n_prompt] * num_samples
            if self.bs == 2 and self.model.clip_trt is not None:
                cond_crossattn, uncond_crossattn = self.model.get_learned_conditioning([cond_prompt, uncond_prompt]).chunk(2)
            else:
                cond_crossattn = self.model.get_learned_conditioning(cond_prompt)
                uncond_crossattn = self.model.get_learned_conditioning(uncond_prompt)
            
            cond = {"c_concat": [control], "c_crossattn": [cond_crossattn]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [uncond_crossattn]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]

        return results
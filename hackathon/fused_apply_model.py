from typing import Any
import numpy as np
import torch
import tensorrt as trt
from cuda import cudart

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)

class FusedControlnetAndUnetTrt(object):
    def __init__(self, controlnet_engine_path, unet_engine_path, cuda_stream, bs=2, use_cuda_graph=True):
        assert bs == 2
        
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.stream = cuda_stream
        
        # UNINITED/UNCAPTURED/READY
        self.cuda_graph_status = "UNINITED"
        self.cuda_graph = None
        self.cuda_graph_exec = None
        self.use_cuda_graph = use_cuda_graph
        self.batch_size = bs
        
        print("Deserializing controlnet_trt engine...")
        trt.init_libnvinfer_plugins(None, "")
        with open(controlnet_engine_path, "rb") as f:
            engine_buf = f.read()
        self.controlnet_engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_buf)
        print("Succeed deserializing controlnet_trt engine!")
        self.controlnet_context = self.controlnet_engine.create_execution_context()

        self.controlnet_nIO = self.controlnet_engine.num_io_tensors
        self.controlnet_lTensorName = [self.controlnet_engine.get_tensor_name(i) for i in range(self.controlnet_nIO)]
        self.controlnet_nInput = [self.controlnet_engine.get_tensor_mode(self.controlnet_lTensorName[i]) for i in range(self.controlnet_nIO)].count(trt.TensorIOMode.INPUT)
        self.controlnet_nOutput = self.controlnet_nIO - self.controlnet_nInput

        for i in range(self.controlnet_nInput):
            shape = self.controlnet_context.get_binding_shape(i)
            shape[0] = bs
            self.controlnet_context.set_binding_shape(i, shape)
        
        self.controlnet_context.infer_shapes()

        for i in range(self.controlnet_nIO):
            print("controlnet_trt [%2d]%s->" % (i, "Input " if i < self.controlnet_nInput else "Output"), self.controlnet_engine.get_tensor_dtype(self.controlnet_lTensorName[i]), self.controlnet_engine.get_tensor_shape(self.controlnet_lTensorName[i]), self.controlnet_context.get_tensor_shape(self.controlnet_lTensorName[i]), self.controlnet_lTensorName[i])

        print("Deserializing unet_trt engine...")
        trt.init_libnvinfer_plugins(None, "")
        with open(unet_engine_path, "rb") as f:
            engine_buf = f.read()
        self.unet_engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_buf)
        print("Succeed deserializing unet_trt engine!")
        self.unet_context = self.unet_engine.create_execution_context()

        self.unet_nIO = self.unet_engine.num_io_tensors
        self.unet_lTensorName = [self.unet_engine.get_tensor_name(i) for i in range(self.unet_nIO)]
        self.unet_nInput = [self.unet_engine.get_tensor_mode(self.unet_lTensorName[i]) for i in range(self.unet_nIO)].count(trt.TensorIOMode.INPUT)
        self.unet_nOutput = self.unet_nIO - self.unet_nInput

        for i in range(self.unet_nInput):
            shape = self.unet_context.get_binding_shape(i)
            shape[0] = bs
            self.unet_context.set_binding_shape(i, shape)
        
        self.unet_context.infer_shapes()

        for i in range(self.unet_nIO):
            print("unet_trt [%2d]%s->" % (i, "Input " if i < self.unet_nInput else "Output"), self.unet_engine.get_tensor_dtype(self.unet_lTensorName[i]), self.unet_engine.get_tensor_shape(self.unet_lTensorName[i]), self.unet_context.get_tensor_shape(self.unet_lTensorName[i]), self.unet_lTensorName[i])

        print("Setting up IO buffers...")
        self.buffersD = {}
        self.buffersT = {}
        self.dTensorInfo = {}
        for i in range(self.controlnet_nIO):
            shape = self.controlnet_context.get_binding_shape(i)
            trt_type = self.controlnet_engine.get_tensor_dtype(self.controlnet_lTensorName[i])
            buf = np.empty(shape, dtype=trt.nptype(trt_type))
            if self.controlnet_lTensorName[i] not in self.dTensorInfo:
                self.dTensorInfo[self.controlnet_lTensorName[i]] = (buf.shape, torch_dtype_from_trt(trt_type), buf.nbytes)
                self.buffersT[self.controlnet_lTensorName[i]] = torch.empty(size=buf.shape, dtype=torch_dtype_from_trt(trt_type)).cuda()
                self.buffersD[self.controlnet_lTensorName[i]] = self.buffersT[self.controlnet_lTensorName[i]].data_ptr()
        for i in range(self.unet_nIO):
            shape = self.unet_context.get_binding_shape(i)
            trt_type = self.unet_engine.get_tensor_dtype(self.unet_lTensorName[i])
            buf = np.empty(shape, dtype=trt.nptype(trt_type))
            if self.unet_lTensorName[i] not in self.dTensorInfo:
                self.dTensorInfo[self.unet_lTensorName[i]] = (buf.shape, torch_dtype_from_trt(trt_type), buf.nbytes)
                self.buffersT[self.unet_lTensorName[i]] = torch.empty(size=buf.shape, dtype=torch_dtype_from_trt(trt_type)).cuda()
                self.buffersD[self.unet_lTensorName[i]] = self.buffersT[self.unet_lTensorName[i]].data_ptr()
        
        for i in range(self.controlnet_nIO):
            self.controlnet_context.set_tensor_address(self.controlnet_lTensorName[i], int(self.buffersD[self.controlnet_lTensorName[i]]))
        for i in range(self.unet_nIO):
            self.unet_context.set_tensor_address(self.unet_lTensorName[i], int(self.buffersD[self.unet_lTensorName[i]]))
        print("Succeed setting up IO buffers!")
        
        if self.use_cuda_graph:
            print("Setting up cuda graph")
            self.execute_inference()
            cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            self.execute_inference()
            status, self.cuda_graph = cudart.cudaStreamEndCapture(self.stream)
            status, self.cuda_graph_exec = cudart.cudaGraphInstantiate(self.cuda_graph, 0)
            cudart.cudaGraphLaunch(self.cuda_graph_exec, self.stream)
            self.cuda_graph_status = "READY"
            print("Successfully captured CUDA Graph!")
        
        print("TRT engine init finished!")

    def execute_inference(self):
        torch.cuda.nvtx.range_push("[fused_applymodel_controlnet_bs2]")
        self.controlnet_context.execute_async_v3(self.stream)
        torch.cuda.nvtx.range_pop()    
        torch.cuda.nvtx.range_push("[fused_applymodel_unet_bs2]")
        self.unet_context.execute_async_v3(self.stream)
        torch.cuda.nvtx.range_pop()

    def fused_apply_model_bs2(self, x_noisy, t, cond, uncond):
        torch.cuda.nvtx.range_push("[prepare_fused_bs2]")
        
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        uncond_txt = torch.cat(uncond['c_crossattn'], 1)
        
        assert cond['c_concat'] is not None
        hint_cond = torch.cat(cond['c_concat'], 1)
        hint_uncond = torch.cat(uncond['c_concat'], 1)
        
        self.buffersT["x_noisy"][:] = torch.cat((x_noisy, x_noisy), 0)
        self.buffersT["timesteps"][:] = torch.cat((t, t), 0).to(torch.int32)
        self.buffersT["context"][:] = torch.cat((cond_txt, uncond_txt), 0)
        self.buffersT["hint"][:] = torch.cat((hint_cond, hint_uncond), 0)

        torch.cuda.nvtx.range_pop()
        
        if self.use_cuda_graph:
            cudart.cudaGraphLaunch(self.cuda_graph_exec, self.stream)
        else:
            self.execute_inference()
        
        # create device tensor for torch
        with torch.device('cuda'):
            outputBuf = torch.empty(size=self.dTensorInfo["eps"][0], dtype=torch.float32)
        # copy outputs
        cudart.cudaMemcpyAsync(outputBuf.data_ptr(), self.buffersD["eps"], self.dTensorInfo["eps"][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
        
        # no sync since we are using same stream for torch
        # cudart.cudaStreamSynchronize(self.stream)
        return outputBuf
    
    def cuda():
        pass

    def deinitialize(self):
        for b in self.buffersD.values():
            cudart.cudaFree(b)

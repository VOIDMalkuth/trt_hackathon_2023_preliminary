from typing import Any
import numpy as np
import torch
import tensorrt as trt
from cuda import cudart

from transformers import CLIPTokenizer

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

class TRTDriver(object):
    def __init__(self, trt_engine_path, bs=1):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        
        print("Deserializing controlnet_trt engine...")
        trt.init_libnvinfer_plugins(None, "")
        with open(trt_engine_path, "rb") as f:
            engine_buf = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_buf)
        print("Succeed deserializing controlnet_trt engine!")
        
        self.batch_size = bs

        self.context = self.engine.create_execution_context()

        self.nIO = self.engine.num_io_tensors
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT)
        self.nOutput = self.nIO - self.nInput

        for i in range(self.nInput):
            shape = self.context.get_binding_shape(i)
            shape[0] = bs
            self.context.set_binding_shape(i, shape)
        
        self.context.infer_shapes()
        
        for i in range(self.nIO):
            print("controlnet_trt [%2d]%s->" % (i, "Input " if i < self.nInput else "Output"), self.engine.get_tensor_dtype(self.lTensorName[i]), self.engine.get_tensor_shape(self.lTensorName[i]), self.context.get_tensor_shape(self.lTensorName[i]), self.lTensorName[i])

        print("Setting up IO buffers...")
        self.buffersD = []
        self.lTensorInfo = []
        for i in range(self.nIO):
            shape = self.context.get_binding_shape(i)
            trt_type = self.engine.get_tensor_dtype(self.lTensorName[i])
            buf = np.empty(shape, dtype=trt.nptype(trt_type))
            self.lTensorInfo.append((buf.shape, torch_dtype_from_trt(trt_type), buf.nbytes))

            self.buffersD.append(cudart.cudaMalloc(buf.nbytes)[1])
        
        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorName[i], int(self.buffersD[i]))
        print("Succeed setting up IO buffers!")

        print("Controlnet_trt engine init finished!")


    def do_inference(self, inputBuffers):
        # copy inputs to device
        # inputBuffers is list of torch cuda tensors
        for i in range(self.nInput):
            cudart.cudaMemcpy(self.buffersD[i], inputBuffers[i].data_ptr(), self.lTensorInfo[i][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        # start execution
        self.context.execute_async_v3(0)
        # create device tensor for torch
        with torch.device('cuda'):
            outputBufs = []
            for i in range(self.nInput, self.nIO):
                outputBufs.append(torch.empty(size=self.lTensorInfo[i][0], dtype=torch.float32))
        # copy outputs
        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpy(outputBufs[i - self.nInput].data_ptr(), self.buffersD[i], self.lTensorInfo[i][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        return outputBufs
    
    def cuda():
        pass

    def deinitialize(self):
        for b in self.buffersD:
            cudart.cudaFree(b)

class TRTDriverCUDAGraph(object):
    def __init__(self, trt_engine_path, bs=1):
        self.logger = trt.Logger(trt.Logger.VERBOSE)

        print("Setup CUDA streams...")
        _, self.stream = cudart.cudaStreamCreate()
        # UNINITED/UNCAPTURED/READY
        self.cuda_graph_status = "UNINITED"
        self.cuda_graph = None
        self.cuda_graph_exec = None
        self.batch_size = bs
        
        print("Deserializing controlnet_trt engine...")
        trt.init_libnvinfer_plugins(None, "")
        with open(trt_engine_path, "rb") as f:
            engine_buf = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_buf)
        print("Succeed deserializing controlnet_trt engine!")

        self.context = self.engine.create_execution_context()

        self.nIO = self.engine.num_io_tensors
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT)
        self.nOutput = self.nIO - self.nInput

        for i in range(self.nInput):
            shape = self.context.get_binding_shape(i)
            shape[0] = bs
            self.context.set_binding_shape(i, shape)
        
        self.context.infer_shapes()

        for i in range(self.nIO):
            print("controlnet_trt [%2d]%s->" % (i, "Input " if i < self.nInput else "Output"), self.engine.get_tensor_dtype(self.lTensorName[i]), self.engine.get_tensor_shape(self.lTensorName[i]), self.context.get_tensor_shape(self.lTensorName[i]), self.lTensorName[i])

        print("Setting up IO buffers...")
        self.buffersD = []
        self.lTensorInfo = []
        for i in range(self.nIO):
            shape = self.context.get_binding_shape(i)
            trt_type = self.engine.get_tensor_dtype(self.lTensorName[i])
            buf = np.empty(shape, dtype=trt.nptype(trt_type))
            self.lTensorInfo.append((buf.shape, torch_dtype_from_trt(trt_type), buf.nbytes))

            self.buffersD.append(cudart.cudaMalloc(buf.nbytes)[1])
        
        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorName[i], int(self.buffersD[i]))
        print("Succeed setting up IO buffers!")
        print("TRT engine init finished!")


    def do_inference(self, inputBuffers):
        # copy inputs to device
        # inputBuffers is list of torch cuda tensors
        for i in range(self.nInput):
            cudart.cudaMemcpyAsync(self.buffersD[i], inputBuffers[i].data_ptr(), self.lTensorInfo[i][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
        
        # start execution
        if self.cuda_graph_status == "UNINITED":
            self.context.execute_async_v3(self.stream)
            self.cuda_graph_status = "UNCAPTURED"
        elif self.cuda_graph_status == "UNCAPTURED":
            cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            self.context.execute_async_v3(self.stream)
            _, self.cuda_graph = cudart.cudaStreamEndCapture(self.stream)
            _, self.cuda_graph_exec = cudart.cudaGraphInstantiate(self.cuda_graph, 0)
            self.cuda_graph_status = "READY"
            print("Successfully captured CUDA Graph!")
        else:
            cudart.cudaGraphLaunch(self.cuda_graph_exec, self.stream)

        # create device tensor for torch
        with torch.device('cuda'):
            outputBufs = []
            for i in range(self.nInput, self.nIO):
                outputBufs.append(torch.empty(size=self.lTensorInfo[i][0], dtype=torch.float32))
        # copy outputs
        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpyAsync(outputBufs[i - self.nInput].data_ptr(), self.buffersD[i], self.lTensorInfo[i][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        return outputBufs
    
    def cuda():
        pass

    def deinitialize(self):
        for b in self.buffersD:
            cudart.cudaFree(b)

class TRTDriverCUDAGraphAsync(object):
    def __init__(self, trt_engine_path, cuda_stream, bs=1, use_cuda_graph=True):
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
        with open(trt_engine_path, "rb") as f:
            engine_buf = f.read()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_buf)
        print("Succeed deserializing controlnet_trt engine!")

        self.context = self.engine.create_execution_context()

        self.nIO = self.engine.num_io_tensors
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.nIO)].count(trt.TensorIOMode.INPUT)
        self.nOutput = self.nIO - self.nInput

        for i in range(self.nInput):
            shape = self.context.get_binding_shape(i)
            shape[0] = bs
            self.context.set_binding_shape(i, shape)
        
        self.context.infer_shapes()

        for i in range(self.nIO):
            print("controlnet_trt [%2d]%s->" % (i, "Input " if i < self.nInput else "Output"), self.engine.get_tensor_dtype(self.lTensorName[i]), self.engine.get_tensor_shape(self.lTensorName[i]), self.context.get_tensor_shape(self.lTensorName[i]), self.lTensorName[i])

        print("Setting up IO buffers...")
        self.buffersD = []
        self.lTensorInfo = []
        for i in range(self.nIO):
            shape = self.context.get_binding_shape(i)
            trt_type = self.engine.get_tensor_dtype(self.lTensorName[i])
            buf = np.empty(shape, dtype=trt.nptype(trt_type))
            self.lTensorInfo.append((buf.shape, torch_dtype_from_trt(trt_type), buf.nbytes))

            self.buffersD.append(cudart.cudaMalloc(buf.nbytes)[1])
        
        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorName[i], int(self.buffersD[i]))
        print("Succeed setting up IO buffers!")
        
        if self.use_cuda_graph:
            print("Setting up cuda graph")
            self.context.execute_async_v3(self.stream)
            cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            self.context.execute_async_v3(self.stream)
            status, self.cuda_graph = cudart.cudaStreamEndCapture(self.stream)
            status, self.cuda_graph_exec = cudart.cudaGraphInstantiate(self.cuda_graph, 0)
            cudart.cudaGraphLaunch(self.cuda_graph_exec, self.stream)
            self.cuda_graph_status = "READY"
            print("Successfully captured CUDA Graph!")
        
        print("TRT engine init finished!")

    def do_inference(self, inputBuffers):
        # copy inputs to device
        # inputBuffers is list of torch cuda tensors
        for i in range(self.nInput):
            cudart.cudaMemcpyAsync(self.buffersD[i], inputBuffers[i].data_ptr(), self.lTensorInfo[i][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
        # create device tensor for torch
        with torch.device('cuda'):
            outputBufs = []
            for i in range(self.nInput, self.nIO):
                outputBufs.append(torch.empty(size=self.lTensorInfo[i][0], dtype=torch.float32))
        
        # start execution
        if self.use_cuda_graph:
            cudart.cudaGraphLaunch(self.cuda_graph_exec, self.stream)
        else:
            self.context.execute_async_v3(self.stream)

        # copy outputs
        for i in range(self.nInput, self.nIO):
            cudart.cudaMemcpyAsync(outputBufs[i - self.nInput].data_ptr(), self.buffersD[i], self.lTensorInfo[i][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
        
        # no sync since we are using same stream for torch
        # cudart.cudaStreamSynchronize(self.stream)
        return outputBufs
    
    def cuda():
        pass

    def deinitialize(self):
        for b in self.buffersD:
            cudart.cudaFree(b)

class ControlNetTRT(TRTDriverCUDAGraphAsync):
    def __call__(self, x=None, hint=None, timesteps=None, context=None) -> Any:
        assert(x is not None)
        assert(hint is not None)
        assert(timesteps is not None)
        assert(context is not None)
        
        inputBuffers = [x.contiguous().cuda(), hint.contiguous().cuda(), timesteps.contiguous().cuda(), context.contiguous().cuda()]

        inference_results = self.do_inference(inputBuffers)
        if len(inference_results) == 1:
            inference_results = inference_results[0]
        return inference_results

class UNetTRT(TRTDriverCUDAGraphAsync):
    def __call__(self, x=None, timesteps=None, context=None, control=None, only_mid_control=False) -> Any:
        assert(not only_mid_control)
        assert(x is not None)
        assert(timesteps is not None)
        assert(context is not None)
        assert(control is not None)
        assert(len(control) == 13)
        
        inputBuffers = [x.contiguous().cuda(), timesteps.contiguous().cuda(), context.contiguous().cuda()]
        for i in range(len(control)):
            inputBuffers.append(control[i].cuda())

        inference_results = self.do_inference(inputBuffers)
        if len(inference_results) == 1:
            inference_results = inference_results[0]
        return inference_results

class VaeTRT(TRTDriverCUDAGraphAsync):
    def __call__(self, z) -> Any:
        inputBuffers = [z.cuda()]

        inference_results = self.do_inference(inputBuffers)
        if len(inference_results) == 1:
            inference_results = inference_results[0]
        return inference_results
    
class ClipTRT(TRTDriverCUDAGraphAsync):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        version="openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
    
    def __call__(self, text) -> Any:
        if self.batch_size == 1:
            batch_encoding = self.tokenizer(text, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(torch.int32).cuda()
        else:
            assert len(text) == 2
            batch_encoding0 = self.tokenizer(text[0], truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens0 = batch_encoding0["input_ids"].to(torch.int32).cuda()
            batch_encoding1 = self.tokenizer(text[1], truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens1 = batch_encoding1["input_ids"].to(torch.int32).cuda()
            tokens = torch.cat((tokens0, tokens1), 0)
        
        inputBuffers = [tokens.contiguous().cuda()]
        inference_results = self.do_inference(inputBuffers)
        return inference_results[0]
    
class FusedTRT(TRTDriverCUDAGraphAsync):
    def fused_apply_model_bs2_simplified(self, x_noisy, context, hint, ts):
        x = torch.cat((x_noisy, x_noisy), 0)
        inputBuffers = [x.contiguous().cuda(), hint.contiguous().cuda(), ts.contiguous().cuda(), context.contiguous().cuda()]
        inference_results = self.do_inference(inputBuffers)
        if len(inference_results) == 1:
            inference_results = inference_results[0]
        return inference_results
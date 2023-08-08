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

class ControlNetTRT(object):
    def __init__(self, trt_engine_path):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        
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

        for i in range(self.nIO):
            print("controlnet_trt [%2d]%s->" % (i, "Input " if i < self.nInput else "Output"), self.engine.get_tensor_dtype(self.lTensorName[i]), self.engine.get_tensor_shape(self.lTensorName[i]), self.context.get_tensor_shape(self.lTensorName[i]), self.lTensorName[i])

        print("Setting up IO buffers...")
        self.buffersD = []
        self.lTensorInfo = []
        for i in range(self.nIO):
            shape = self.context.get_tensor_shape(self.lTensorName[i])
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
    
    def __call__(self, x=None, hint=None, timesteps=None, context=None) -> Any:
        assert(x is not None)
        assert(hint is not None)
        assert(timesteps is not None)
        assert(context is not None)
        
        inputBuffers = [x.cuda(), hint.cuda(), timesteps.cuda(), context.cuda()]

        inference_results = self.do_inference(inputBuffers)
        if len(inference_results) == 1:
            inference_results = inference_results[0]
        return inference_results

    def deinitialize(self):
        for b in self.buffersD:
            cudart.cudaFree(b)

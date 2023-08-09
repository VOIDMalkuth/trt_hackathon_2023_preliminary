# export onnx models
rm -rf onnx_models
mkdir -p onnx_models
mkdir -p onnx_models/controlnet
mkdir -p onnx_models/unet
PYTHONPATH=$PWD python3 hackathon/tools/export_onnx.py

onnxsim onnx_models/controlnet/controlnet_static_shape.onnx onnx_models/controlnet/controlnet_static_shape.onnx &> /dev/null
onnxsim onnx_models/unet/unet_static_shape.onnx onnx_models/unet/unet_static_shape.onnx &> /dev/null

echo "building trt_controlnet"
python3 hackathon/tools/build_controlnet_trt_engine.py

echo "building trt_unet"
python3 hackathon/tools/build_unet_trt_engine.py

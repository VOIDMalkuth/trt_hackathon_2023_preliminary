echo "preprocess start"

# no internet so not possible
# echo "installing onnx-simplifier"
# pip install onnx-simplifier

echo "exporting onnx models"
rm -rf onnx_models
mkdir -p onnx_models
mkdir -p onnx_models/controlnet
mkdir -p onnx_models/unet
PYTHONPATH=$PWD python3 hackathon/tools/export_onnx.py

# no internet so not possible
# echo "simplifying onnx models with onnx-simplifier"
# onnxsim onnx_models/controlnet/controlnet_static_shape.onnx onnx_models/controlnet/controlnet_static_shape.onnx
# onnxsim onnx_models/unet/unet_static_shape.onnx onnx_models/unet/unet_static_shape.onnx

echo "build trt_engine for controlnet"
python3 hackathon/tools/build_controlnet_trt_engine.py

echo "build trt_engine for unet"
python3 hackathon/tools/build_unet_trt_engine.py

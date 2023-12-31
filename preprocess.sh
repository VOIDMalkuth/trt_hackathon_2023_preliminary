# export onnx models
rm -rf onnx_models
mkdir -p onnx_models
mkdir -p onnx_models/controlnet
mkdir -p onnx_models/unet
mkdir -p onnx_models/vae
mkdir -p onnx_models/clip
mkdir -p onnx_models/fused_model

export ON_CLOUD_TEST=1

echo "Exporting onnx"
PYTHONPATH=$PWD python3 hackathon/tools/export_onnx.py

echo "Optimizing onnx"
# onnxsim onnx_models/controlnet/controlnet_static_shape.onnx onnx_models/controlnet/controlnet_static_shape.onnx
# onnxsim onnx_models/unet/unet_static_shape.onnx onnx_models/unet/unet_static_shape.onnx
onnxsim onnx_models/fused_model/fused_model_static_shape.onnx onnx_models/fused_model/fused_model_static_shape.onnx
onnxsim onnx_models/vae/vae_static_shape.onnx onnx_models/vae/vae_static_shape.onnx
onnxsim onnx_models/clip/clip_static_shape.onnx onnx_models/clip/clip_static_shape.onnx

# echo "building trt_controlnet"
# python3 hackathon/tools/build_controlnet_trt_engine.py

# echo "building trt_unet"
# python3 hackathon/tools/build_unet_trt_engine.py

echo "building trt_fused_model"
python3 hackathon/tools/build_fused_model_trt_engine.py

echo "building trt_vae"
python3 hackathon/tools/build_vae_trt_engine.py

echo "building trt_clip"
python3 hackathon/tools/build_clip_trt_engine.py
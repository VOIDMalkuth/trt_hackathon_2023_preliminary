import tensorrt as trt
from cuda import cudart

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

def build_controlnet_trt_engine():
    ONNX_FILE_PATH = "onnx_models/controlnet/controlnet_static_shape.onnx"
    TRT_ENGINE_PATH = "trt_controlnet_batch_1.plan"

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # modif config
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    # config.builder_optimization_level = 3
    config.set_flag(trt.BuilderFlag.FP16)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_FILE_PATH, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)
    print("Succeeded parsing .onnx file!")

    input_x_noisy = network.get_input(0)
    input_hint = network.get_input(1)
    input_timesteps = network.get_input(2)
    input_context = network.get_input(3)
    profile.set_shape(input_x_noisy.name, X_NOISY_SHAPE, X_NOISY_SHAPE, X_NOISY_SHAPE)
    profile.set_shape(input_hint.name, IMAGE_HINT_SHAPE, IMAGE_HINT_SHAPE, IMAGE_HINT_SHAPE)
    profile.set_shape(input_timesteps.name, TIMESTEPS_SHAPE, TIMESTEPS_SHAPE, TIMESTEPS_SHAPE)
    profile.set_shape(input_context.name, CONTEXT_SHAPE, CONTEXT_SHAPE, CONTEXT_SHAPE)

    config.add_optimization_profile(profile)

    engine_buf = builder.build_serialized_network(network, config)
    if engine_buf == None:
        print("failed to build engine!")
        exit(-1)
    print("Succeeded building engine!")

    with open(TRT_ENGINE_PATH, "wb") as f:
        f.write(engine_buf)

    print("Succeeded building and serializing trt engine for controlnet!")

if __name__ == "__main__":
    build_controlnet_trt_engine()
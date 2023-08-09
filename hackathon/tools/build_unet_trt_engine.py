import os
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

def build_unet_trt_engine():
    ONNX_WEIGHT_DIR = "onnx_models/unet/"
    ONNX_FILE_PATH = "unet_static_shape.onnx"
    TRT_ENGINE_PATH = "trt_unet_batch_1.plan"
    TIME_CACHE_FILE_PATH = "time_cache.dat"

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # modif config
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    # config.builder_optimization_level = 3
    config.set_flag(trt.BuilderFlag.FP16)

    # read time_cache
    time_cache_bytes = b""
    if os.path.isfile(TIME_CACHE_FILE_PATH):
        with open(TIME_CACHE_FILE_PATH, "rb") as f:
            time_cache = f.read()
    time_cache = config.create_timing_cache(time_cache_bytes)
    config.set_timing_cache(time_cache, False)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    parser = trt.OnnxParser(network, logger)

    # change to ONNX_WEIGHT_DIR for weight data reading
    cwd = os.getcwd()
    os.chdir(ONNX_WEIGHT_DIR)
    with open(ONNX_FILE_PATH, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)
    print("Succeeded parsing .onnx file!")
    os.chdir(cwd)

    input_x = network.get_input(0)
    input_timesteps = network.get_input(1)
    input_context = network.get_input(2)
    profile.set_shape(input_x.name, X_NOISY_SHAPE, X_NOISY_SHAPE, X_NOISY_SHAPE)
    profile.set_shape(input_timesteps.name, TIMESTEPS_SHAPE, TIMESTEPS_SHAPE, TIMESTEPS_SHAPE)
    profile.set_shape(input_context.name, CONTEXT_SHAPE, CONTEXT_SHAPE, CONTEXT_SHAPE)
    for i in range(len(CONTROL_FEATURE_SHAPES)):
        input_control_i = network.get_input(3 + i)
        profile.set_shape(input_control_i.name, CONTROL_FEATURE_SHAPES[i], CONTROL_FEATURE_SHAPES[i], CONTROL_FEATURE_SHAPES[i])

    config.add_optimization_profile(profile)

    engine_buf = builder.build_serialized_network(network, config)
    if engine_buf == None:
        print("failed to build engine!")
        exit(-1)
    print("Succeeded building engine!")

    with open(TRT_ENGINE_PATH, "wb") as f:
        f.write(engine_buf)

    # write timing cache
    time_cache = config.get_timing_cache()
    time_cache_bytes = time_cache.serialize()
    with open(TIME_CACHE_FILE_PATH, "wb") as f:
        f.write(time_cache_bytes)

    print("Succeeded building and serializing trt engine for unet!")

if __name__ == "__main__":
    build_unet_trt_engine()
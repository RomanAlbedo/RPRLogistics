#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime_c_api.h>
#include "stb_image.h"
#include "stb_image_resize.h"

#define CHECK(status) if (status != NULL) { \
    fprintf(stderr, "ONNXRuntime Error: %s\n", OrtGetErrorMessage(status)); \
    exit(1); \
}

// Normalize input [0–255] → [0–1]
void preprocess(float* out, unsigned char* in, int w, int h) {
    for (int i = 0; i < w * h; ++i) {
        out[i + 0*w*h] = in[i*3 + 0] / 255.0f;
        out[i + 1*w*h] = in[i*3 + 1] / 255.0f;
        out[i + 2*w*h] = in[i*3 + 2] / 255.0f;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s model.onnx input.jpg output.raw width height\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];
    const char* output_path = argv[3];
    int W = atoi(argv[4]);
    int H = atoi(argv[5]);

    // Load and resize image
    int iw, ih, ch;
    unsigned char* img = stbi_load(image_path, &iw, &ih, &ch, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }

    unsigned char* resized = malloc(W * H * 3);
    stbir_resize_uint8(img, iw, ih, 0, resized, W, H, 0, 3);
    stbi_image_free(img);

    float* input_data = malloc(sizeof(float) * 3 * W * H);
    preprocess(input_data, resized, W, H);
    free(resized);

    // Initialize ONNX Runtime
    OrtEnv* env;
    CHECK(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "inference", &env));

    OrtSessionOptions* opts;
    CHECK(OrtCreateSessionOptions(&opts));

    OrtSession* session;
    CHECK(OrtCreateSession(env, model_path, opts, &session));

    // Get input/output names
    OrtAllocator* allocator;
    CHECK(OrtCreateAllocator(env, OrtArenaAllocator, NULL, &allocator));

    char* input_name;
    CHECK(OrtSessionGetInputName(session, 0, allocator, &input_name));

    char* output_name;
    CHECK(OrtSessionGetOutputName(session, 0, allocator, &output_name));

    printf("Input Name: %s\n", input_name);
    printf("Output Name: %s\n", output_name);

    // Prepare input tensor
    int64_t input_shape[] = {1, 3, H, W};
    OrtMemoryInfo* mem_info;
    CHECK(OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    OrtValue* input_tensor;
    CHECK(OrtCreateTensorWithDataAsOrtValue(mem_info, input_data, sizeof(float) * 3 * H * W,
                                            input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                            &input_tensor));

    // Run inference
    OrtValue* output_tensor = NULL;
    CHECK(OrtRun(session, NULL,
                 (const char* const*)&input_name, (const OrtValue* const*)&input_tensor, 1,
                 (const char* const*)&output_name, 1, &output_tensor));

    // Get output shape
    OrtTensorTypeAndShapeInfo* output_info;
    CHECK(OrtGetTensorTypeAndShape(output_tensor, &output_info));

    size_t out_num_dims;
    CHECK(OrtGetDimensionsCount(output_info, &out_num_dims));

    int64_t* out_shape = malloc(sizeof(int64_t) * out_num_dims);
    CHECK(OrtGetDimensions(output_info, out_shape, out_num_dims));

    size_t out_size = 1;
    for (size_t i = 0; i < out_num_dims; ++i)
        out_size *= out_shape[i];

    printf("Output Shape: [");
    for (size_t i = 0; i < out_num_dims; ++i)
        printf("%lld%s", out_shape[i], (i < out_num_dims - 1) ? ", " : "]\n");

    // Access output data
    float* output_data;
    CHECK(OrtGetTensorMutableData(output_tensor, (void**)&output_data));

    // Save output to file
    FILE* f = fopen(output_path, "wb");
    fwrite(output_data, sizeof(float), out_size, f);
    fclose(f);

    // Cleanup
    free(out_shape);
    free(input_data);
    OrtReleaseValue(input_tensor);
    OrtReleaseValue(output_tensor);
    OrtReleaseTensorTypeAndShapeInfo(output_info);
    OrtReleaseMemoryInfo(mem_info);
    OrtReleaseSession(session);
    OrtReleaseSessionOptions(opts);
    OrtReleaseAllocator(allocator);
    OrtReleaseEnv(env);

    return 0;
}

include(FetchContent)

FetchContent_Declare(
    cuda_samples
    GIT_REPOSITORY https://github.com/NVIDIA/cuda-samples.git
    GIT_TAG v13.1
    SYSTEM
    EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(cuda_samples)

# Set variables for use in the sorting benchmark
set(CUDA_SAMPLES_INCLUDE_DIRS "${cuda_samples_SOURCE_DIR}/Common" "${cuda_samples_SOURCE_DIR}/Samples")
set(CUDA_SAMPLES_FOUND TRUE)

message(STATUS "CUDA samples downloaded to: ${cuda_samples_SOURCE_DIR}")

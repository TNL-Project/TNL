tnl_add_test_executables(UniquePointerTest SOURCES UniquePointerTest.cpp BACKENDS HOST)

# tests with different sources for each backend
tnl_add_test_executables(
    SharedPointerTest
    SOURCES SharedPointerTest.cpp
    CUDA_SOURCES SharedPointerTestCuda.cu
    HIP_SOURCES SharedPointerTestHip.hip
)
tnl_add_test_executables(DevicePointerTest CUDA_SOURCES DevicePointerTestCuda.cu HIP_SOURCES DevicePointerTestHip.hip)

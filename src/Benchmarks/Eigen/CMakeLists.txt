set(BENCHMARK_NAMES tnl-benchmark-eigen-qr tnl-benchmark-eigen-pi tnl-benchmark-eigen-random)

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
foreach(target IN ITEMS ${BENCHMARK_NAMES})
    tnl_add_backend_executables(${target} SOURCES ${target}.h BACKENDS CUDA HIP HOST OPTIONAL_HOST)
endforeach()

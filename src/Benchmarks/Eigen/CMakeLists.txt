set(BENCHMARK_NAMES tnl-benchmark-eigen-qr tnl-benchmark-eigen-pi tnl-benchmark-eigen-random)

# built only for one backend (GPU if available)
foreach(target IN ITEMS ${BENCHMARK_NAMES})
    tnl_add_backend_executables(${target} SOURCES ${target}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

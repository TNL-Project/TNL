set(NDARRAY_BENCHMARKS
    tnl-benchmark-ndarray
    tnl-benchmark-ndarray-boundary
    tnl-benchmark-reduction
    tnl-benchmark-ndarray-reduction
)

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
foreach(bench IN LISTS NDARRAY_BENCHMARKS)
    tnl_add_backend_executables(
        ${bench}
        SOURCES ${bench}.h
        BACKENDS HOST CUDA HIP
        OPTIONAL_HOST
        OUTPUT_TARGETS benchmark_targets
    )
    install(TARGETS ${benchmark_targets} RUNTIME DESTINATION bin COMPONENT benchmarks)
endforeach()

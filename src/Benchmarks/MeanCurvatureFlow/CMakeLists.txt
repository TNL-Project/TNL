# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-mean-curvature-flow
    SOURCES tnl-benchmark-mean-curvature-flow.h
    BACKENDS CUDA HIP HOST
    OPTIONAL_HOST
    OUTPUT_TARGETS BENCHMARKS
    OUTPUT_BACKENDS BENCHMARK_BACKENDS
)
tnl_benchmark_link_comparison_libraries(TARGETS ${BENCHMARKS} BACKENDS ${BENCHMARK_BACKENDS})

install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)

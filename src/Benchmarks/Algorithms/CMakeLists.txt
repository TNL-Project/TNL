tnl_add_backend_executables(
    tnl-benchmark-parallel-for
    SOURCES tnl-benchmark-parallel-for.h
    BACKENDS CUDA HIP HOST
    OPTIONAL_HOST
    OUTPUT_TARGETS BENCHMARKS
)

install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-memory-access
    SOURCES tnl-benchmark-memory-access.h
    BACKENDS HOST
    OPTIONAL_HOST
    OUTPUT_TARGETS BENCHMARKS
)

if(BENCHMARKS)
    install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

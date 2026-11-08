# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(tnl-benchmark-segments SOURCES tnl-benchmark-segments.h OPTIONAL_HOST OUTPUT_TARGETS BENCHMARKS)

if(BENCHMARKS)
    install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

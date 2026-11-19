tnl_add_backend_executables(
    tnl-benchmark-grid-register-pressure
    SOURCES tnl-benchmark-grid-register-pressure.h
    BACKENDS CUDA HIP HOST
    OPTIONAL_HOST
    OUTPUT_TARGETS BENCHMARKS
)

install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)

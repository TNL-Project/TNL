set(GRID_DIMS 1 2 3)

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
foreach(DIM IN ITEMS ${GRID_DIMS})
    tnl_add_backend_executables(
        tnl-benchmark-grid-${DIM}D
        SOURCES tnl-benchmark-grid.h
        BACKENDS CUDA HIP HOST
        OPTIONAL_HOST
        COMPILE_DEFINITIONS GRID_DIM=${DIM}
        OUTPUT_TARGETS benchmark_targets
    )
    install(TARGETS ${benchmark_targets} RUNTIME DESTINATION bin COMPONENT benchmarks)
endforeach()

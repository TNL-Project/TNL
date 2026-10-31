set(GRID_DIMS 1 2 3)

# built only for one backend (GPU if available)
foreach(DIM IN ITEMS ${GRID_DIMS})
    tnl_add_backend_executables(
        tnl-benchmark-grid-${DIM}D
        SOURCES tnl-benchmark-grid.h
        BACKENDS CUDA HIP HOST
        SINGLE
        COMPILE_DEFINITIONS GRID_DIM=${DIM}
    )
    install(TARGETS tnl-benchmark-grid-${DIM}D RUNTIME DESTINATION bin COMPONENT benchmarks)
endforeach()

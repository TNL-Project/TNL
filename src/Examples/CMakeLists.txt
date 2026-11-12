add_subdirectory(Ginkgo)
add_subdirectory(Hypre)

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-turbulence-generator
    SOURCES tnl-turbulence-generator.cpp
    BACKENDS HOST CUDA HIP
    OPTIONAL_HOST
    OUTPUT_TARGETS targets
)

set(MPI_OPTIONS)
if(TNL_BUILD_MPI)
    # enable MPI support in TNL and add MPI to the targets:
    # https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
    set(MPI_OPTIONS COMPILE_DEFINITIONS HAVE_MPI LINK_LIBRARIES MPI::MPI_CXX)
endif()
tnl_add_backend_executables(
    tnl-optimize-ranks
    SOURCES tnl-optimize-ranks.cpp
    BACKENDS HOST CUDA HIP
    OPTIONAL_HOST
    ${MPI_OPTIONS}
    OUTPUT_TARGETS OPTIMIZE_RANKS_TARGETS
)
list(APPEND targets ${OPTIMIZE_RANKS_TARGETS})

if(targets)
    install(TARGETS ${targets} RUNTIME DESTINATION bin COMPONENT examples)
endif()

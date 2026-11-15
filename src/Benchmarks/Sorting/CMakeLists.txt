if(TNL_BUILD_CUDA)
    find_package(CCCL REQUIRED)
endif()

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-sort
    SOURCES tnl-benchmark-sort.h
    OPTIONAL_HOST
    CUDA_LINK_LIBRARIES CCCL::CCCL
    OUTPUT_TARGETS BENCHMARKS
)

# Add include directories from CUDA samples for reference algorithms
if(TARGET tnl-benchmark-sort-cuda AND CUDA_SAMPLES_FOUND)
    target_include_directories(tnl-benchmark-sort-cuda SYSTEM PRIVATE ${CUDA_SAMPLES_INCLUDE_DIRS})
    target_compile_definitions(tnl-benchmark-sort-cuda PRIVATE HAVE_CUDA_SAMPLES)
endif()

# OPTIONAL_HOST skips building host-only targets in CUDA-enabled CI jobs
tnl_add_backend_executables(
    tnl-benchmark-pivot-partition
    SOURCES tnl-benchmark-pivot-partition.h
    OPTIONAL_HOST
    OUTPUT_TARGETS PIVOT_PARTITION_BENCHMARKS
)

list(APPEND BENCHMARKS ${PIVOT_PARTITION_BENCHMARKS})

if(BENCHMARKS)
    # reference CPU algorithms (ips2ra, ips4o); their parallel variants need
    # TBB and threads (-pthread defines _REENTRANT, which enables them)
    if(TARGET tnl-benchmark-sort AND SORTING_REFERENCES_FOUND)
        find_package(TBB QUIET)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads)
        if(TBB_FOUND AND Threads_FOUND)
            target_include_directories(tnl-benchmark-sort SYSTEM PRIVATE ${SORTING_REFERENCES_INCLUDE_DIRS})
            target_compile_definitions(tnl-benchmark-sort PRIVATE HAVE_SORTING_REFERENCES)
            target_link_libraries(tnl-benchmark-sort PRIVATE TBB::tbb Threads::Threads)
        else()
            message(WARNING "TBB was not found - the reference sorting algorithms (ips2ra, ips4o) are disabled.")
        endif()
    endif()
    install(TARGETS ${BENCHMARKS} RUNTIME DESTINATION bin COMPONENT benchmarks)
endif()

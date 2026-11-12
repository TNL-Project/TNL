add_subdirectory(Algorithms)
add_subdirectory(Containers)
add_subdirectory(Graphs)
add_subdirectory(Pointers)
add_subdirectory(Matrices)
add_subdirectory(Meshes)
add_subdirectory(Solvers)

set(COMMON_EXAMPLES)

set(GPU_EXAMPLES FileExampleCuda)

set(HOST_EXAMPLES
    FileExample
    FileExampleSaveAndLoad
    ParseObjectTypeExample
    PerformanceCountersExample
    StringExample
    StringExampleGetAllocatedSize
    StringExampleReplace
    StringExampleSplit
    StringExampleSetSize
    StringExampleStrip
    TimerExample
    TimerExampleLogger
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h)
endforeach()

foreach(target IN ITEMS ${GPU_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.h BACKENDS CUDA HIP)
endforeach()

foreach(target IN ITEMS ${HOST_EXAMPLES})
    # PerformanceCountersExample example requires admin privileges to run on macOS so we only build it
    # on the Apple platform but do not execute it.
    if(APPLE AND target STREQUAL "PerformanceCountersExample")
        tnl_add_backend_executables(${target} SOURCES ${target}.cpp BACKENDS HOST)
    else()
        tnl_add_example_with_output(${target} SOURCES ${target}.cpp BACKENDS HOST)
    endif()
endforeach()

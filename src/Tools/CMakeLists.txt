# Mesh tools: will link to zlib and tinyxml2
set(MESH_TOOLS
    tnl-decompose-grid
    tnl-grid-to-mesh
    tnl-mesh-converter
    tnl-mesh-info
    tnl-triangulate-mesh
    tnl-planar-correct-mesh
    tnl-refine-mesh
    tnl-game-of-life
    tnl-reorder-mesh
    tnl-plot-mesh-ordering
)

# CGAL tools: will link to CGAL
set(CGAL_TOOLS tnl-reorder-mesh tnl-plot-mesh-ordering)

# MPI tools: will link to MPI
set(MPI_TOOLS tnl-game-of-life)

# skip building host-only targets in CUDA-enabled CI jobs
if(NOT TNL_BUILD_CPP_TARGETS)
    set(MESH_TOOLS)
    set(MPI_TOOLS)
    set(CGAL_TOOLS)
endif()

foreach(target IN ITEMS ${MESH_TOOLS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PUBLIC TNL::TNL)
endforeach()

# mesh tool with MPI built also for CUDA and HIP
tnl_add_backend_executables(
    tnl-test-distributed-mesh
    SOURCES tnl-test-distributed-mesh.h
    BACKENDS HOST CUDA HIP
    OPTIONAL_HOST
    OUTPUT_TARGETS DISTRIBUTED_MESH_TOOLS
)
list(APPEND MESH_TOOLS ${DISTRIBUTED_MESH_TOOLS})
list(APPEND MPI_TOOLS ${DISTRIBUTED_MESH_TOOLS})

find_package(ZLIB)
find_package(tinyxml2)
foreach(target IN ITEMS ${MESH_TOOLS})
    if(ZLIB_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
        target_link_libraries(${target} PUBLIC ZLIB::ZLIB)
    endif()
    if(tinyxml2_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
        target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)
    endif()
    install(TARGETS ${target} DESTINATION bin COMPONENT tools)
endforeach()

if(TNL_BUILD_MPI)
    foreach(target IN ITEMS ${MPI_TOOLS})
        # enable MPI support in TNL
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endforeach()
endif()

set(CGAL_DATA_DIR "/") # to disable stupid warning: https://github.com/CGAL/cgal/pull/6649
find_package(CGAL)
if(CGAL_FOUND)
    option(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE "disable warning" ON)
    foreach(target IN ITEMS ${CGAL_TOOLS})
        target_compile_definitions(${target} PUBLIC "-DHAVE_CGAL")
        target_link_libraries(${target} PUBLIC CGAL::CGAL)
    endforeach()
endif()

# skip building host-only targets in CUDA-enabled CI jobs
find_package(METIS QUIET)
if(TNL_BUILD_CPP_TARGETS AND METIS_FOUND)
    add_executable(tnl-decompose-mesh tnl-decompose-mesh.cpp)
    target_link_libraries(tnl-decompose-mesh PUBLIC TNL::TNL METIS::METIS)
    if(ZLIB_FOUND)
        target_compile_definitions(tnl-decompose-mesh PUBLIC "-DHAVE_ZLIB")
        target_link_libraries(tnl-decompose-mesh PUBLIC ZLIB::ZLIB)
    endif()
    if(tinyxml2_FOUND)
        target_compile_definitions(tnl-decompose-mesh PUBLIC "-DHAVE_TINYXML2")
        target_link_libraries(tnl-decompose-mesh PUBLIC tinyxml2::tinyxml2)
    endif()
    install(TARGETS tnl-decompose-mesh DESTINATION bin COMPONENT tools)
endif()

if(TNL_BUILD_CUDA)
    add_executable(tnl-cuda-arch tnl-cuda-arch.cu)
    install(TARGETS tnl-cuda-arch DESTINATION bin COMPONENT tools)
endif()

# skip building host-only targets in CUDA-enabled CI jobs
if(TNL_BUILD_CPP_TARGETS)
    add_executable(tnl-matrix-info tnl-matrix-info.cpp)
    target_link_libraries(tnl-matrix-info PUBLIC TNL::TNL)

    add_executable(tnl-grid-setup tnl-grid-setup.cpp)
    target_link_libraries(tnl-grid-setup PUBLIC TNL::TNL)

    install(TARGETS tnl-matrix-info tnl-grid-setup DESTINATION bin COMPONENT tools)

    install(PROGRAMS tnl-benchmark-to-html.py tnl-run-benchmarks.py DESTINATION bin COMPONENT tools)
endif()

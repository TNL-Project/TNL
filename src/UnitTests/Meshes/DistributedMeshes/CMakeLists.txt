add_executable(DirectionsTest DirectionsTest.cpp)
target_link_libraries(DirectionsTest PUBLIC TNL::TNL TNL_tests)

add_executable(CopyEntitiesTest CopyEntitiesTest.cpp)
target_link_libraries(CopyEntitiesTest PUBLIC TNL::TNL TNL_tests)

# TODO: Fix distributed grid.
#ADD_EXECUTABLE( CutMeshFunctionTest CutMeshFunctionTest.cpp )
#   TARGET_LINK_LIBRARIES( CutMeshFunctionTest PUBLIC TNL::TNL TNL_tests )

tnl_add_test(NAME DirectionsTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DirectionsTest${CMAKE_EXECUTABLE_SUFFIX})
tnl_add_test(NAME CopyEntitiesTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CopyEntitiesTest${CMAKE_EXECUTABLE_SUFFIX})
# ADD_TEST( NAME CutMeshFunctionTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX} )

if(TNL_BUILD_MPI)
    # TODO: Fix distributed grid.
    #ADD_EXECUTABLE( DistributedGridTest_1D DistributedGridTest_1D.cpp )
    #TARGET_LINK_LIBRARIES( DistributedGridTest_1D PUBLIC TNL::TNL TNL_tests )

    #ADD_EXECUTABLE( DistributedGridTest_2D DistributedGridTest_2D.cpp )
    #TARGET_LINK_LIBRARIES( DistributedGridTest_2D PUBLIC TNL::TNL TNL_tests )

    #ADD_EXECUTABLE( DistributedGridTest_3D DistributedGridTest_3D.cpp )
    #TARGET_LINK_LIBRARIES( DistributedGridTest_3D PUBLIC TNL::TNL TNL_tests )

    #ADD_EXECUTABLE( CutDistributedGridTest CutDistributedGridTest.cpp )
    #TARGET_LINK_LIBRARIES( CutDistributedGridTest PUBLIC TNL::TNL TNL_tests )

    #ADD_EXECUTABLE( CutDistributedMeshFunctionTest CutDistributedMeshFunctionTest.cpp )
    #TARGET_LINK_LIBRARIES( CutDistributedMeshFunctionTest PUBLIC TNL::TNL TNL_tests )

    # built only for one backend (GPU if available), the tests are registered
    # below with different numbers of processes, so they are not added with
    # tnl_add_test_executables
    tnl_add_backend_executables(
        DistributedMeshTest
        SOURCES DistributedMeshTest.h ../../main_mpi.h
        BACKENDS CUDA HIP HOST
        SINGLE
        LINK_LIBRARIES TNL_tests
    )

    # TODO: Fix distributed grid.
    #foreach( target IN ITEMS DistributedGridTest_1D DistributedGridTest_2D DistributedGridTest_3D CutDistributedGridTest DistributedMeshTest )
    foreach(target IN ITEMS DistributedMeshTest)
        # enable MPI support in TNL
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endforeach()

    # external libraries for tests which use mesh readers
    find_package(ZLIB)
    find_package(tinyxml2)

    if(ZLIB_FOUND AND tinyxml2_FOUND)
        # TODO: Fix distributed grid.
        # foreach( target IN ITEMS DistributedGridTest_1D DistributedGridTest_2D DistributedGridTest_3D DistributedMeshTest )
        foreach(target IN ITEMS DistributedMeshTest)
            target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
            target_link_libraries(${target} PUBLIC ZLIB::ZLIB)

            target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
            target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)
        endforeach()
    endif()

    # TODO: Fix distributed grid
    #tnl_add_mpi_test( NAME DistributedGridTest_1D NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_1D${CMAKE_EXECUTABLE_SUFFIX}" )
    #tnl_add_mpi_test( NAME DistributedGridTest_2D NPROC 9 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_2D${CMAKE_EXECUTABLE_SUFFIX}" )
    #tnl_add_mpi_test( NAME DistributedGridTest_3D NPROC 27 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_3D${CMAKE_EXECUTABLE_SUFFIX}" )

    #tnl_add_mpi_test( NAME CutDistributedGridTest NPROC 12 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}" )

    # TODO: Fix
    #tnl_add_mpi_test( NAME CutDistributedMeshFunctionTest NPROC 12 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX}" )

    if(TNL_BUILD_HIP)
        tnl_add_hip_mpi_test(
            NAME DistributedMeshTest_2x2
            NPROC 4
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
        tnl_add_hip_mpi_test(
            NAME DistributedMeshTest_3x3
            NPROC 9
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
        tnl_add_hip_mpi_test(
            NAME DistributedMeshTest_4x4
            NPROC 16
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
        tnl_add_hip_mpi_test(
            NAME DistributedMeshTest_nodistr
            NPROC 1
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
    else()
        tnl_add_mpi_test(
            NAME DistributedMeshTest_2x2
            NPROC 4
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
        tnl_add_mpi_test(
            NAME DistributedMeshTest_3x3
            NPROC 9
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
        tnl_add_mpi_test(
            NAME DistributedMeshTest_4x4
            NPROC 16
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endif()
endif()

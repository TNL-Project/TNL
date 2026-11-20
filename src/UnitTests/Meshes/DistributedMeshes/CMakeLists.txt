add_executable(DirectionsTest DirectionsTest.cpp)
target_link_libraries(DirectionsTest PUBLIC TNL::TNL TNL_tests)

add_executable(CopyEntitiesTest CopyEntitiesTest.cpp)
target_link_libraries(CopyEntitiesTest PUBLIC TNL::TNL TNL_tests)

tnl_add_test(NAME DirectionsTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DirectionsTest${CMAKE_EXECUTABLE_SUFFIX})
tnl_add_test(NAME CopyEntitiesTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CopyEntitiesTest${CMAKE_EXECUTABLE_SUFFIX})

if(TNL_BUILD_MPI)
    # built only for one backend (GPU if available), the tests are registered
    # below with different numbers of processes, so they are not added with
    # tnl_add_test_executables
    tnl_add_backend_executables(
        DistributedGridTest
        SOURCES DistributedGridTest.h ../../main_mpi.h
        BACKENDS CUDA HIP HOST
        SINGLE
        LINK_LIBRARIES TNL_tests
    )
    tnl_add_backend_executables(
        DistributedMeshTest
        SOURCES DistributedMeshTest.h ../../main_mpi.h
        BACKENDS CUDA HIP HOST
        SINGLE
        LINK_LIBRARIES TNL_tests
    )

    foreach(target IN ITEMS DistributedGridTest DistributedMeshTest)
        # enable MPI support in TNL
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endforeach()

    # external libraries for tests which use mesh readers
    find_package(ZLIB)
    find_package(tinyxml2)

    if(ZLIB_FOUND AND tinyxml2_FOUND)
        foreach(target IN ITEMS DistributedMeshTest)
            target_compile_definitions(${target} PUBLIC "-DHAVE_ZLIB")
            target_link_libraries(${target} PUBLIC ZLIB::ZLIB)

            target_compile_definitions(${target} PUBLIC "-DHAVE_TINYXML2")
            target_link_libraries(${target} PUBLIC tinyxml2::tinyxml2)
        endforeach()
    endif()

    foreach(nproc IN ITEMS 1 2 4 9)
        if(TNL_BUILD_HIP)
            tnl_add_hip_mpi_test(
                NAME DistributedGridTest_${nproc}
                NPROC ${nproc}
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}"
            )
        else()
            tnl_add_mpi_test(
                NAME DistributedGridTest_${nproc}
                NPROC ${nproc}
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}"
            )
        endif()
    endforeach()

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

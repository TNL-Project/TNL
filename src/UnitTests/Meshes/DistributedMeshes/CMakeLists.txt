add_executable(DirectionsTest DirectionsTest.cpp)
target_compile_options(DirectionsTest PUBLIC ${CXX_TESTS_FLAGS})
target_link_libraries(DirectionsTest PUBLIC TNL::TNL ${TESTS_LIBRARIES})
target_link_options(DirectionsTest PUBLIC ${TESTS_LINKER_FLAGS})

add_executable(CopyEntitesTest CopyEntitiesTest.cpp)
target_compile_options(CopyEntitesTest PUBLIC ${CXX_TESTS_FLAGS})
target_link_libraries(CopyEntitesTest PUBLIC TNL::TNL ${TESTS_LIBRARIES})
target_link_options(CopyEntitesTest PUBLIC ${TESTS_LINKER_FLAGS})

# TODO: Fix distributed grid.
#ADD_EXECUTABLE( CutMeshFunctionTest CutMeshFunctionTest.cpp )
#   TARGET_COMPILE_OPTIONS( CutMeshFunctionTest PUBLIC ${CXX_TESTS_FLAGS} )
#   TARGET_LINK_LIBRARIES( CutMeshFunctionTest PUBLIC TNL::TNL ${TESTS_LIBRARIES} )
#   target_link_options( CutMeshFunctionTest PUBLIC ${TESTS_LINKER_FLAGS} )

add_test(NAME DirectionsTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DirectionsTest${CMAKE_EXECUTABLE_SUFFIX})
add_test(NAME CopyEntitesTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CopyEntitesTest${CMAKE_EXECUTABLE_SUFFIX})
# ADD_TEST( NAME CutMeshFunctionTest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX} )

if(TNL_BUILD_MPI)
    # TODO: Fix distributed grid.
    #ADD_EXECUTABLE( DistributedGridTest_1D DistributedGridTest_1D.cpp )
    #TARGET_COMPILE_OPTIONS( DistributedGridTest_1D PUBLIC ${CXX_TESTS_FLAGS} )
    #TARGET_LINK_LIBRARIES( DistributedGridTest_1D PUBLIC TNL::TNL ${TESTS_LIBRARIES} )
    #target_link_options( DistributedGridTest_1D PUBLIC ${TESTS_LINKER_FLAGS} )

    #ADD_EXECUTABLE( DistributedGridTest_2D DistributedGridTest_2D.cpp )
    #TARGET_COMPILE_OPTIONS( DistributedGridTest_2D PUBLIC ${CXX_TESTS_FLAGS} )
    #TARGET_LINK_LIBRARIES( DistributedGridTest_2D PUBLIC TNL::TNL ${TESTS_LIBRARIES} )
    #target_link_options( DistributedGridTest_2D PUBLIC ${TESTS_LINKER_FLAGS} )

    #ADD_EXECUTABLE( DistributedGridTest_3D DistributedGridTest_3D.cpp )
    #TARGET_COMPILE_OPTIONS( DistributedGridTest_3D PUBLIC ${CXX_TESTS_FLAGS} )
    #TARGET_LINK_LIBRARIES( DistributedGridTest_3D PUBLIC TNL::TNL ${TESTS_LIBRARIES} )
    #target_link_options( DistributedGridTest_3D PUBLIC ${TESTS_LINKER_FLAGS} )

    #ADD_EXECUTABLE( CutDistributedGridTest CutDistributedGridTest.cpp )
    #TARGET_COMPILE_OPTIONS( CutDistributedGridTest PUBLIC ${CXX_TESTS_FLAGS} )
    #TARGET_LINK_LIBRARIES( CutDistributedGridTest PUBLIC TNL::TNL ${TESTS_LIBRARIES} )
    #target_link_options( CutDistributedGridTest PUBLIC ${TESTS_LINKER_FLAGS} )

    #ADD_EXECUTABLE( CutDistributedMeshFunctionTest CutDistributedMeshFunctionTest.cpp )
    #TARGET_COMPILE_OPTIONS( CutDistributedMeshFunctionTest PUBLIC ${CXX_TESTS_FLAGS} )
    #TARGET_LINK_LIBRARIES( CutDistributedMeshFunctionTest PUBLIC TNL::TNL ${TESTS_LIBRARIES} )
    #target_link_options( CutDistributedMeshFunctionTest PUBLIC ${TESTS_LINKER_FLAGS} )

    if(TNL_BUILD_CUDA)
        add_executable(DistributedMeshTest DistributedMeshTest.cu)
        target_compile_options(DistributedMeshTest PUBLIC ${CUDA_TESTS_FLAGS})
    elseif(TNL_BUILD_HIP)
        add_executable(DistributedMeshTest DistributedMeshTest.hip)
        target_compile_options(DistributedMeshTest PUBLIC ${HIP_TESTS_FLAGS})
    else()
        add_executable(DistributedMeshTest DistributedMeshTest.cpp)
        target_compile_options(DistributedMeshTest PUBLIC ${CXX_TESTS_FLAGS})
    endif()
    target_link_libraries(DistributedMeshTest PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(DistributedMeshTest PUBLIC ${TESTS_LINKER_FLAGS})

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
    #add_test_mpi( NAME DistributedGridTest_1D NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_1D${CMAKE_EXECUTABLE_SUFFIX}" )
    #add_test_mpi( NAME DistributedGridTest_2D NPROC 9 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_2D${CMAKE_EXECUTABLE_SUFFIX}" )
    #add_test_mpi( NAME DistributedGridTest_3D NPROC 27 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedGridTest_3D${CMAKE_EXECUTABLE_SUFFIX}" )

    #add_test_mpi( NAME CutDistributedGridTest NPROC 12 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedGridTest${CMAKE_EXECUTABLE_SUFFIX}" )

    # TODO: Fix
    #add_test_mpi( NAME CutDistributedMeshFunctionTest NPROC 12 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/CutDistributedMeshFunctionTest${CMAKE_EXECUTABLE_SUFFIX}" )

    add_test_mpi(
        NAME DistributedMeshTest_2x2 NPROC 4 COMMAND
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
    )
    # FIXME: 3x3 and 4x4 tests cause a deadlock on a single AMD Radeon RX 7900 GRE
    if(NOT TNL_BUILD_HIP)
        add_test_mpi(
            NAME DistributedMeshTest_3x3 NPROC 9 COMMAND
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
        add_test_mpi(
            NAME DistributedMeshTest_4x4 NPROC 16 COMMAND
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DistributedMeshTest${CMAKE_EXECUTABLE_SUFFIX}"
        )
    endif()

endif()

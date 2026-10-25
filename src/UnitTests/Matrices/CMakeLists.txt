# formats of segments for which the tests of the reduction, traversal and vector product of sparse matrices are
# instantiated (see ../Algorithms/Segments/SegmentsTestFormats.h and SparseMatrixTestTypes.h)
set(SPARSE_MATRIX_TEST_FORMATS
    CSR
    AdaptiveCSR
    Ellpack
    SlicedEllpack
    ChunkedEllpack
    BiEllpack
)
set(SPARSE_MATRIX_TEST_TYPES
    "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
    TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
    TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long > >"
)

add_subdirectory(Factorization)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

# generates the SparseMatrixTest executables for all formats of segments including the sorted ones,
# with all variants of each format (e.g. row-major and column-major)
tnl_add_format_tests(
    NAME SparseMatrixTest
    SUITE MatrixTest
    PREFIX SparseMatrix<FORMAT>
    TYPES
        "TNL::UnitTests::SparseMatrixVariantsTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
        TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrixTestTypes.h ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrixTest.h
    FORMATS
        ${SPARSE_MATRIX_TEST_FORMATS}
        SortedCSR
        SortedAdaptiveCSR
        SortedEllpack
        SortedSlicedEllpack
        SortedChunkedEllpack
        SortedBiEllpack
    SPLIT
        CSR:3
        AdaptiveCSR:3
        Ellpack:5
        SlicedEllpack:5
        ChunkedEllpack:5
        BiEllpack:5
        SortedCSR:3
        SortedAdaptiveCSR:3
        SortedEllpack:5
        SortedSlicedEllpack:5
        SortedChunkedEllpack:5
        SortedBiEllpack:5
    LABELS Matrices
)

# generates the SparseMatrixVectorProductTest executables for all formats of segments in
# SPARSE_MATRIX_TEST_FORMATS, with all variants of each format (e.g. row-major and column-major)
tnl_add_format_tests(
    NAME SparseMatrixVectorProductTest
    SUITE MatrixTest
    PREFIX SparseMatrix<FORMAT>
    TYPES
        "TNL::UnitTests::SparseMatrixVariantsTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
        TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrixTestTypes.h ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrixVectorProductTest.h
    FORMATS ${SPARSE_MATRIX_TEST_FORMATS}
    LABELS Matrices
)

# generates the BinarySparseMatrixTest executables for the formats of segments supported by
# binary sparse matrices
tnl_add_format_tests(
    NAME BinarySparseMatrixTest
    SUITE BinaryMatrixTest
    PREFIX BinarySparseMatrix<FORMAT>
    TYPES
        "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< bool, int, int >, TNL::UnitTests::MatrixElementTypes< bool, long, int > >"
    HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/SparseMatrixTestTypes.h ${CMAKE_CURRENT_SOURCE_DIR}/BinarySparseMatrixTestSuite.hpp
    FORMATS CSR AdaptiveCSR Ellpack SlicedEllpack
    LABELS Matrices
)

set(COMMON_TESTS
    DenseMatrixTest
    DenseMatrixCopyTest
    TridiagonalMatrixTest
    MultidiagonalMatrixTest
    SparseDenseMatrixOperationsTest_CSR
    SparseMatrixOperationsTest_CSR
    SparseMatrixCopyTest
    BinarySparseMatrixCopyTest
    SymmetricSparseMatrixTest_CSR
    LambdaMatrixTest
    MatrixOperationsTest
    MatrixWriterTest
    MatrixWrappingTest
    DenseMatrixMultiplicationTest
    DenseMatrixTranspositionTest
    DenseMatrixTransMultiplication
)

set(CPP_TESTS ${COMMON_TESTS} StaticMatrixTest MatrixTypeTraitsTest MatrixReaderTest)
set(CUDA_TESTS)
set(HIP_TESTS)

foreach(target IN ITEMS ${COMMON_TESTS})
    set(CUDA_TESTS ${CUDA_TESTS} ${target}_cuda)
    set(HIP_TESTS ${HIP_TESTS} ${target}_hip)
endforeach()

set(MPI_TESTS DistributedMatrixTest)
set(MPI_CUDA_TESTS DistributedMatrixTest_cuda)
set(MPI_HIP_TESTS DistributedMatrixTest_hip)

foreach(target IN ITEMS ${CPP_TESTS})
    add_executable(${target} ${target}.cpp)
    target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
    target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
    target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
    add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
    set_property(TEST ${target} APPEND PROPERTY LABELS Matrices)
endforeach()

if(TNL_BUILD_CUDA)
    foreach(target IN ITEMS ${CUDA_TESTS})
        string(REGEX REPLACE "_cuda$" "" common_target ${target})
        tnl_test_sources(sources NAME ${common_target} SOURCE ${target}.cu SPLIT_TESTS ${SPLIT_TESTS})
        add_executable(${target} ${sources})
        target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        add_test(${target} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX})
        # add "Matrices" label to the test
        set_property(TEST ${target} APPEND PROPERTY LABELS Matrices)
    endforeach()
endif()

if(TNL_BUILD_HIP)
    foreach(target IN ITEMS ${HIP_TESTS})
        string(REGEX REPLACE "_hip$" "" common_target ${target})
        tnl_test_sources(sources NAME ${common_target} SOURCE ${target}.hip SPLIT_TESTS ${SPLIT_TESTS})
        add_executable(${target} ${sources})
        target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        tnl_add_hip_test(NAME ${target} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        # add "Matrices" label to the test
        set_property(TEST ${target} APPEND PROPERTY LABELS Matrices)
    endforeach()
endif()

if(TNL_BUILD_MPI)
    foreach(target IN ITEMS ${MPI_TESTS})
        add_executable(${target} ${target}.cpp)
        target_compile_options(${target} PUBLIC ${CXX_TESTS_FLAGS})
        target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
        target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
        tnl_add_mpi_test(NAME ${target} NPROC 4 COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        tnl_add_mpi_test(
            NAME ${target}_nodistr
            NPROC 1
            COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
        )
        # add "Matrices" label to the test
        set_property(TEST ${target} APPEND PROPERTY LABELS Matrices)
        set_property(TEST ${target}_nodistr APPEND PROPERTY LABELS Matrices)
    endforeach()

    if(TNL_BUILD_CUDA)
        foreach(target IN ITEMS ${MPI_CUDA_TESTS})
            add_executable(${target} ${target}.cu)
            target_compile_options(${target} PUBLIC ${CUDA_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
            tnl_add_mpi_test(
                NAME ${target}
                NPROC 4
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            tnl_add_mpi_test(
                NAME ${target}_nodistr
                NPROC 1
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            # add "Matrices" label to the test
            set_property(TEST ${target} APPEND PROPERTY LABELS Matrices)
            set_property(TEST ${target}_nodistr APPEND PROPERTY LABELS Matrices)
        endforeach()
    endif()

    if(TNL_BUILD_HIP)
        foreach(target IN ITEMS ${MPI_HIP_TESTS})
            add_executable(${target} ${target}.hip)
            target_compile_options(${target} PUBLIC ${HIP_TESTS_FLAGS})
            target_link_libraries(${target} PUBLIC TNL::TNL ${TESTS_LIBRARIES})
            target_link_options(${target} PUBLIC ${TESTS_LINKER_FLAGS})
            target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
            target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
            tnl_add_hip_mpi_test(
                NAME ${target}
                NPROC 4
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            tnl_add_hip_mpi_test(
                NAME ${target}_nodistr
                NPROC 1
                COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}${CMAKE_EXECUTABLE_SUFFIX}"
            )
            # add "Matrices" label to the test
            set_property(TEST ${target}_nodistr APPEND PROPERTY LABELS Matrices)
        endforeach()
    endif()
endif()

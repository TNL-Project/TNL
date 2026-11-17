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
# covering set of the element types for the GPU builds (see ../ARCHITECTURE.md)
set(SPARSE_MATRIX_TEST_GPU_TYPES
    "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
    TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< double, long > >"
)

add_subdirectory(Factorization)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

# generates the SparseMatrixVectorProductTest executables for all formats of segments in
# SPARSE_MATRIX_TEST_FORMATS, with all variants of each format (e.g. row-major and column-major)
tnl_add_format_tests(
    SparseMatrixVectorProductTest
    SUITE MatrixTest
    PREFIX SparseMatrix<FORMAT>
    TYPES
        "TNL::UnitTests::SparseMatrixVariantsTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
        TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    GPU_TYPES
        "TNL::UnitTests::SparseMatrixVariantsCoveringTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    HEADERS SparseMatrixTestTypes.h SparseMatrixVectorProductTest.h
    FORMATS ${SPARSE_MATRIX_TEST_FORMATS}
    LABELS Matrices
)

# generates the BinarySparseMatrixTest executables for the formats of segments supported by
# binary sparse matrices
tnl_add_format_tests(
    BinarySparseMatrixTest
    SUITE BinaryMatrixTest
    PREFIX BinarySparseMatrix<FORMAT>
    TYPES
        "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<FORMAT>Format, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< bool, int, int >, TNL::UnitTests::MatrixElementTypes< bool, long, int > >"
    HEADERS SparseMatrixTestTypes.h BinarySparseMatrixTestSuite.hpp
    FORMATS CSR AdaptiveCSR Ellpack SlicedEllpack
    LABELS Matrices
)

set(TESTS
    DenseMatrixTest
    DenseMatrixCopyTest
    TridiagonalMatrixTest
    MultidiagonalMatrixTest
    SparseDenseMatrixOperationsTest_CSR
    SparseMatrixOperationsTest_CSR
    SparseMatrixTest_CSR
    SparseMatrixTest_AdaptiveCSR
    SparseMatrixTest_Ellpack
    SparseMatrixTest_SlicedEllpack
    SparseMatrixTest_ChunkedEllpack
    SparseMatrixTest_BiEllpack
    SparseMatrixTest_SortedAdaptiveCSR
    SparseMatrixTest_SortedCSR
    SparseMatrixTest_SortedBiEllpack
    SparseMatrixTest_SortedChunkedEllpack
    SparseMatrixTest_SortedEllpack
    SparseMatrixTest_SortedSlicedEllpack
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

set(HOST_TESTS StaticMatrixTest MatrixTypeTraitsTest MatrixReaderTest)
set(MPI_TESTS DistributedMatrixTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h LABELS Matrices)
endforeach()

foreach(test IN ITEMS ${HOST_TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.cpp BACKENDS HOST LABELS Matrices)
endforeach()

foreach(test IN ITEMS ${MPI_TESTS})
    tnl_add_test_executables(${test} MPI SOURCES ${test}.h LABELS Matrices)
endforeach()

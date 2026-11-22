# segments types for which the tests of the reduction, traversal and vector product of sparse matrices are
# instantiated (see ../Algorithms/Segments/SegmentsTestVariants.h and SparseMatrixTestTypes.h)
set(SPARSE_MATRIX_TESTED_SEGMENTS_TYPES
    CSR
    AdaptiveCSR
    Ellpack
    SlicedEllpack
    ChunkedEllpack
    BiEllpack
)
set(SPARSE_MATRIX_TEST_TYPES
    "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants, TNL::UnitTests::TestDevice,
    TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
    TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long > >"
)
# covering set of the element types for the GPU builds (see ../ARCHITECTURE.md)
set(SPARSE_MATRIX_TEST_GPU_TYPES
    "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants, TNL::UnitTests::TestDevice,
    TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< double, long > >"
)

add_subdirectory(Factorization)
add_subdirectory(Reduce)
add_subdirectory(Traverse)

# generates the SparseMatrixTest executables for all segments types including the sorted ones,
# with all variants of each segments type (e.g. row-major and column-major)
tnl_add_segments_tests(
    SparseMatrixTest
    SUITE MatrixTest
    PREFIX SparseMatrix<SEGMENTS_TYPE>
    TYPES
        "TNL::UnitTests::SparseMatrixVariantsTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
        TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    GPU_TYPES
        "TNL::UnitTests::SparseMatrixVariantsCoveringTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants,
        TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
        TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    HEADERS SparseMatrixTestTypes.h SparseMatrixTest.h
    SEGMENTS_TYPES
        ${SPARSE_MATRIX_TESTED_SEGMENTS_TYPES}
        SortedCSR
        SortedAdaptiveCSR
        SortedEllpack
        SortedSlicedEllpack
        SortedChunkedEllpack
        SortedBiEllpack
    LABELS Matrices
)

# generates the SparseMatrixVectorProductTest executables for all segments types in
# SPARSE_MATRIX_TESTED_SEGMENTS_TYPES, with all variants of each segments type (e.g. row-major and column-major)
tnl_add_segments_tests(
    SparseMatrixVectorProductTest
    SUITE MatrixTest
    PREFIX SparseMatrix<SEGMENTS_TYPE>
    TYPES
        "TNL::UnitTests::SparseMatrixVariantsTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< long, long >,
        TNL::UnitTests::MatrixElementTypes< float, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    GPU_TYPES
        "TNL::UnitTests::SparseMatrixVariantsCoveringTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants,
        TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< int, int >, TNL::UnitTests::MatrixElementTypes< double, long >,
        TNL::UnitTests::MatrixElementTypes< TNL::Arithmetics::Complex< float >, long > >"
    HEADERS SparseMatrixTestTypes.h SparseMatrixVectorProductTest.h
    SEGMENTS_TYPES ${SPARSE_MATRIX_TESTED_SEGMENTS_TYPES}
    LABELS Matrices
)

# generates the BinarySparseMatrixTest executables for the segments types supported by
# binary sparse matrices
tnl_add_segments_tests(
    BinarySparseMatrixTest
    SUITE BinaryMatrixTest
    PREFIX BinarySparseMatrix<SEGMENTS_TYPE>
    TYPES
        "TNL::UnitTests::SparseMatrixTestTypes< TNL::UnitTests::<SEGMENTS_TYPE>Variants, TNL::UnitTests::TestDevice,
        TNL::UnitTests::MatrixElementTypes< bool, int, int >, TNL::UnitTests::MatrixElementTypes< bool, long, int > >"
    HEADERS SparseMatrixTestTypes.h BinarySparseMatrixTestSuite.hpp
    SEGMENTS_TYPES CSR AdaptiveCSR Ellpack SlicedEllpack
    LABELS Matrices
)

set(TESTS
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

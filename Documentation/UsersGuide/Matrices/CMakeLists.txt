set(COMMON_EXAMPLES
    BinarySparseMatrixExample
    DenseMatrixExample_reduceRows_maxNorm
    DenseMatrixExample_reduceRows_vectorProduct
    DenseMatrixViewExample_data_encapsulation
    SparseMatrixExample_reduceRows_vectorProduct
    SymmetricSparseMatrixExample
)

####
# The following examples/benchmarks run for very long time
# We just build them and do not run automatically.
set(LONG_EXAMPLES DenseMatrixSetup_Benchmark MultidiagonalMatrixSetup_Benchmark SparseMatrixSetup_Benchmark)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()

# The following examples/benchmarks run for a very long time, so we just
# build them and do not run them automatically.
foreach(target IN ITEMS ${LONG_EXAMPLES})
    tnl_add_backend_executables(${target} SOURCES ${target}.cpp BACKENDS CUDA HOST SINGLE)
endforeach()

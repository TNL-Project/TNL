# tests built only for one backend (GPU if available)
set(TESTS Grid1DTest_Traverser Grid2DTest_Traverser Grid3DTest_Traverser GridCoordinateIteratorTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

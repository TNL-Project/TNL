# tests built only for one backend (GPU if available)
set(TESTS Grid1DTest_Accessors Grid2DTest_Accessors Grid3DTest_Accessors)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

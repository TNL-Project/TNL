# tests built only for one backend (GPU if available)
set(TESTS
    Grid1DTest_Neighbours_0D
    Grid1DTest_Neighbours_1D
    Grid2DTest_Neighbours_0D
    Grid2DTest_Neighbours_1D
    Grid2DTest_Neighbours_2D
    Grid3DTest_Neighbours_0D
    Grid3DTest_Neighbours_1D
    Grid3DTest_Neighbours_2D
    Grid3DTest_Neighbours_3D
)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

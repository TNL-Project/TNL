# tests built only for the host backend
set(TESTS Grid1DTest_Accessors Grid2DTest_Accessors Grid3DTest_Accessors)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS HOST)
endforeach()

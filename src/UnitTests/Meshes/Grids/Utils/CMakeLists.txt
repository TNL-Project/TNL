# tests built only for one backend (GPU if available)
set(TESTS CoordinateIteratorTest EntityDataStoreTest GridTemplatesTest)

foreach(test IN ITEMS ${TESTS})
    tnl_add_test_executables(${test} SOURCES ${test}.h ../../../main.h BACKENDS CUDA HIP HOST SINGLE)
endforeach()

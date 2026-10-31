foreach(test IN ITEMS PowerMethodTest QRAlgorithmTest)
    tnl_add_test_executables(${test} SOURCES ${test}.h BACKENDS HOST CUDA)
endforeach()

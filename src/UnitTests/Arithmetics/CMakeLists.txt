tnl_add_test_executables(ComplexTest SOURCES ComplexTest.h BACKENDS HOST)

if(HAVE_GMP)
    add_executable(MultiPrecisionTest MultiPrecisionTest.cpp ../MultiPrecision.cpp)
    target_link_libraries(MultiPrecisionTest PUBLIC TNL::TNL ${GMP_LIBRARIES} TNL_tests)
    tnl_add_test(MultiPrecisionTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/MultiPrecisionTest${CMAKE_EXECUTABLE_SUFFIX})
endif(HAVE_GMP)

# TODO: Fix the following tests
#ADD_EXECUTABLE( QuadTest QuadTest.cpp )
#TARGET_LINK_LIBRARIES( QuadTest PUBLIC TNL::TNL TNL_tests )
#ADD_TEST( QuadTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/QuadTest${CMAKE_EXECUTABLE_SUFFIX} )

#ADD_EXECUTABLE( DoubleTest DoubleTest.cpp )
#TARGET_LINK_LIBRARIES( DoubleTest PUBLIC TNL::TNL TNL_tests )
#ADD_TEST( DoubleTest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DoubleTest${CMAKE_EXECUTABLE_SUFFIX} )

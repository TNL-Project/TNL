tnl_add_test_executables(BubbleSortTest SOURCES BubbleSortTest.cpp BACKENDS HOST)

if(TNL_BUILD_CUDA)
    find_package(CCCL REQUIRED)
elseif(TNL_BUILD_HIP)
    find_package(hipcub REQUIRED)
endif()

# built only for one backend (GPU if available)
tnl_add_test_executables(
    isSortedTest
    SOURCES isSortedTest.h
    BACKENDS CUDA HIP HOST
    SINGLE
    CUDA_LINK_LIBRARIES CCCL::CCCL
    HIP_LINK_LIBRARIES hip::hipcub
)

tnl_add_test_executables(
    SorterTest
    SOURCES SorterTest.h
    BACKENDS CUDA HIP
    SINGLE
    CUDA_LINK_LIBRARIES CCCL::CCCL
    HIP_LINK_LIBRARIES hip::hipcub
)

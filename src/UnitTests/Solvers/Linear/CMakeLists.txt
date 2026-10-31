# built only for one backend (GPU if available)
tnl_add_test_executables(GEMSolverTest SOURCES GEMSolverTest.h BACKENDS CUDA HIP HOST SINGLE)

tnl_add_test_executables(UmfpackWrapperTest SOURCES UmfpackWrapperTest.cpp BACKENDS HOST)

if(NOT TARGET SuiteSparse::UMFPACK)
    find_package(UMFPACK GLOBAL)
endif()
if(TARGET SuiteSparse::UMFPACK)
    target_compile_definitions(UmfpackWrapperTest PUBLIC "-DHAVE_UMFPACK")
    target_link_libraries(UmfpackWrapperTest PUBLIC SuiteSparse::UMFPACK)
    # reset SuiteSparse::UMFPACK to system target (upstream sets 0)
    set_property(TARGET SuiteSparse::UMFPACK PROPERTY SYSTEM 1)
endif()

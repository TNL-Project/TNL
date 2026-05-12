include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    OVERRIDE_FIND_PACKAGE
    SYSTEM
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Prevent installing GTest along with TNL (the tests themselves are not installed)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Fix for Intel oneAPI >= 2025.2: -Wno-implicit-float-size-conversion was removed,
# replaced by -Wno-sycl-implicit-float-size-conversion. GTest adds the old flag for
# IntelLLVM, causing a fatal error with -Werror. Override the compile flags here.
# Upstream PR: https://github.com/google/googletest/pull/4798
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 2025.2.0)
    foreach(_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
        if(TARGET ${_gtest_target})
            get_target_property(_flags ${_gtest_target} COMPILE_FLAGS)
            string(
                REPLACE "-Wno-implicit-float-size-conversion"
                "-Wno-sycl-implicit-float-size-conversion"
                _fixed_flags
                "${_flags}"
            )
            set_target_properties(${_gtest_target} PROPERTIES COMPILE_FLAGS "${_fixed_flags}")
        endif()
    endforeach()
endif()

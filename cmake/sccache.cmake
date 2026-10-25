# Use sccache as the compiler launcher when it is available. sccache keys its entries on the
# preprocessed source and the compiler, so unchanged translation units are reused across
# rebuilds, branches and machines without any per-project cache key.
#
# The storage backend is configured entirely through the SCCACHE_* environment variables
# (SCCACHE_DIR for the local disk cache, SCCACHE_BUCKET and friends for a shared one), so
# nothing has to be set here for a developer's local cache to work.
#
# Only a top-level build is affected. When TNL is used as a subdirectory, the compile-heavy
# targets belong to the parent project and live in a different directory scope anyway.

if(NOT TNL_USE_SCCACHE OR NOT PROJECT_IS_TOP_LEVEL)
    return()
endif()

find_program(TNL_SCCACHE_EXECUTABLE NAMES sccache)
if(NOT TNL_SCCACHE_EXECUTABLE)
    message(STATUS "sccache not found, the compiler cache is disabled")
    return()
endif()

# respect a launcher that was configured explicitly, e.g. ccache in the .env file
foreach(language IN ITEMS C CXX CUDA HIP)
    if(DEFINED CMAKE_${language}_COMPILER_LAUNCHER)
        message(STATUS "compiler launcher already set for ${language}, not using sccache")
        return()
    endif()
endforeach()

message(STATUS "using sccache as the compiler launcher: ${TNL_SCCACHE_EXECUTABLE}")
set(TNL_SCCACHE_LANGUAGES C CXX CUDA HIP)
# sccache treats nvc as a GCC-like compiler and appends -fdiagnostics-color=always, which the
# NVHPC C compiler rejects (nvc++ accepts it), so leave C sources out of the cache
if(CMAKE_C_COMPILER_ID STREQUAL "NVHPC")
    list(REMOVE_ITEM TNL_SCCACHE_LANGUAGES C)
endif()
foreach(language IN ITEMS ${TNL_SCCACHE_LANGUAGES})
    set(CMAKE_${language}_COMPILER_LAUNCHER "${TNL_SCCACHE_EXECUTABLE}")
endforeach()

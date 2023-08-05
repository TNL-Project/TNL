# set default build options
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=vla" )
   set( CMAKE_CXX_FLAGS_DEBUG "-g" )
   set( CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" )
endif()
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}" )

# warn about redundant semicolons
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi" )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra-semi-stmt" )
   endif()
endif()

# disable GCC's infamous "maybe-uninitialized" warning (it produces mostly false positives)
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized" )
endif()

# disable false compiler warnings for NVHPC - see the cuda_flags.cmake file
target_compile_options( TNL_CXX INTERFACE
      $<$<CXX_COMPILER_ID:NVHPC>:
            --diag_suppress=code_is_unreachable ;
            --diag_suppress=loop_not_reachable ;
            --diag_suppress=implicit_return_from_non_void_function ;
            --diag_suppress=unsigned_compare_with_zero ;
            --display_error_number ;
      >
)

if( TNL_USE_CI_FLAGS AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" )
   # enforce (more or less) warning-free builds
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated -Wno-error=deprecated-declarations" )
endif()

if( CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
   # icpx complains about -g without -O
   set( CMAKE_CXX_FLAGS_DEBUG "-O0 -g" )
   # avoid warning: explicit comparison with NaN in fast floating point mode [-Wtautological-constant-compare]
   # see https://github.com/mfem/mfem/issues/3655#issuecomment-1569294763
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fhonor-infinities -fhonor-nans" )
endif()

# optimize Release builds for the native CPU arch, unless explicitly disabled
if( TNL_USE_MARCH_NATIVE_FLAG )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0 )
      # -march=native does not work on macOS with clang less than 15.0
      # https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1
      set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=apple-m1" )
   elseif( CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" )
      set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xhost" )
   else()
      set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native" )
   endif()
endif()

# hack needed to avoid "file too big" errors on a 32-bit cross-compiler
# https://stackoverflow.com/questions/71875002/file-too-big-with-mingw-w64-and-cmake
if( MINGW )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
   endif()
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       OR ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC" )
      )
      set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2" )
   endif()
endif()
if( MSVC )
   # "/bigobj" avoids problems with large object files due to heavily templated code
   # "/permissive-" enables two-phase name lookup according to the standard, see https://stackoverflow.com/q/74238513
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /permissive-" )
endif()

# enable sanitizers (does not work with MPI due to many false positives, does not work with nvcc at all)
# sanitizers are not available for Windows: https://github.com/msys2/MINGW-packages/issues/3163
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
   if( NOT WIN32 AND NOT TNL_USE_MPI AND NOT TNL_USE_CUDA )
      set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer" )
      set( CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG "${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
      set( CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
      set( CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined" )
   endif()
endif()

# enable link time optimizations (but not in continuous integration)
if( NOT DEFINED ENV{GITLAB_CI} )
   if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
      # FIXME: IPO with GCC 9.1.0 and Debug build = internal compiler error
      # FIXME: IPO with GCC 9.1.0 and nvcc 10.1 and Release build = fatal error: bytecode stream in file `blabla` generated with LTO version 7.1 instead of the expected 8.0
      #set( CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE )
   elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
      # FIXME: IPO with clang from MSYS2 leads to 'file format not recognized' linker errors
      # FIXME: clang does not support IPO for CUDA
      #set( CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE )
   endif()
endif()

# force colorized output (the automatic detection in compilers does not work with Ninja)
target_compile_options( TNL_CXX INTERFACE
      $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics> ;
      $<$<CXX_COMPILER_ID:AppleClang>:-fcolor-diagnostics> ;
      $<$<CXX_COMPILER_ID:IntelLLVM>:-fcolor-diagnostics> ;
      $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color> ;
      $<$<CXX_COMPILER_ID:NVHPC>:-fdiagnostics-color> ;
)

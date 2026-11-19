// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Functions.h>

#ifdef TNL_BENCHMARKS_HAVE_KOKKOS
   #include <Kokkos_Core.hpp>
#endif

#ifdef TNL_BENCHMARKS_HAVE_AMREX
   #include <AMReX.H>
   #include <AMReX_ParmParse.H>
#endif

namespace TNL::Benchmarks {

/**
 * \brief Initializes and finalizes the third-party libraries used for the comparison in the grid benchmarks.
 *
 * Kokkos and AMReX are initialized only if the benchmark is built with them (see `TNL_BENCHMARKS_USE_KOKKOS`
 * and `TNL_BENCHMARKS_USE_AMREX` in `src/Benchmarks/CMakeLists.txt`). The object must be created after the
 * TNL device setup, so that the libraries use the same GPU as TNL, and it must outlive all Kokkos views and
 * AMReX data.
 */
class ComparisonLibrariesGuard
{
public:
   ComparisonLibrariesGuard( const char* programName )
   {
#if defined( __CUDACC__ ) || defined( __HIP__ )
      [[maybe_unused]] const int device = Backend::getDevice();
#else
      [[maybe_unused]] const int device = -1;
#endif

#ifdef TNL_BENCHMARKS_HAVE_KOKKOS
      Kokkos::InitializationSettings settings;
      if( device >= 0 )
         settings.set_device_id( device );
      Kokkos::initialize( settings );
#endif

#ifdef TNL_BENCHMARKS_HAVE_AMREX
      // AMReX does not parse the command line of the benchmark, the parameters are set here
      int argc = 1;
      char* argvStorage[] = { const_cast< char* >( programName ), nullptr };
      char** argv = argvStorage;
      auto setParameters = []()
      {
         amrex::ParmParse pp( "amrex" );
         pp.add( "verbose", 0 );
         // do not override the signal handlers of the benchmark
         pp.add( "signal_handling", 0 );
         // by default, AMReX allocates 3/4 of the GPU memory in advance
         pp.add( "the_arena_init_size", static_cast< amrex::Long >( 0 ) );
      };
      amrex::Initialize( argc, argv, false, MPI_COMM_WORLD, setParameters, std::cout, std::cerr, nullptr, device );
#endif
   }

   ComparisonLibrariesGuard( const ComparisonLibrariesGuard& ) = delete;

   ComparisonLibrariesGuard&
   operator=( const ComparisonLibrariesGuard& ) = delete;

   ~ComparisonLibrariesGuard()
   {
#ifdef TNL_BENCHMARKS_HAVE_AMREX
      amrex::Finalize();
#endif
#ifdef TNL_BENCHMARKS_HAVE_KOKKOS
      Kokkos::finalize();
#endif
   }
};

}  // namespace TNL::Benchmarks

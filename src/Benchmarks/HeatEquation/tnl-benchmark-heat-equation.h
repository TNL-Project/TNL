// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

#include "HeatEquationSolverBenchmarkFDMParallelFor.h"
#include "HeatEquationSolverBenchmarkFDMNDArray.h"
#include "HeatEquationSolverBenchmarkFDMGrid.h"
#include "HeatEquationSolverBenchmarkFVMParallelFor.h"
#include "HeatEquationSolverBenchmarkFVMGrid.h"
#include "HeatEquationSolverBenchmarkKokkos.h"
#include "HeatEquationSolverBenchmarkAMReX.h"
#include "HeatEquationSolverBenchmarkGridTools.h"
#include "../GridLibraries/ComparisonLibraries.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "Heat equation benchmark settings:" );
   config.addEntry< TNL::String >( "implementation", "Implementation of the heat equation solver.", "grid" );
   config.addEntryEnum< TNL::String >( "parallel-for" );
   config.addEntryEnum< TNL::String >( "nd-array" );
   config.addEntryEnum< TNL::String >( "grid" );
   config.addEntryEnum< TNL::String >( "kokkos" );
   config.addEntryEnum< TNL::String >( "amrex" );
   config.addEntryEnum< TNL::String >( "gridtools" );
   config.addEntry< TNL::String >( "scheme", "Numerical scheme used for the discretization.", "fdm" );
   config.addEntryEnum< TNL::String >( "fdm" );
   config.addEntryEnum< TNL::String >( "fvm" );

   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );

   config.addEntry< TNL::String >( "device", "Device the computation will run on.", "all" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );

   HeatEquationSolverBenchmark<>::configSetup( config );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
}

/**
 * \brief Runs the Kokkos implementation of the given scheme, if Kokkos supports the device in this build.
 */
template< int Dimension, bool Fvm, typename Real, typename Device >
bool
runKokkos( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
#ifdef TNL_BENCHMARKS_HAVE_KOKKOS
   if constexpr( TNL::Benchmarks::KokkosSupport::isSupported< Device >() ) {
      HeatEquationSolverBenchmarkKokkos< Dimension, Fvm, Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   else {
      std::cerr << "The Kokkos implementation is not available for the device " << TNL::Benchmarks::getDeviceName< Device >()
                << " in this build, skipping it.\n";
      return true;
   }
#else
   std::cerr << "The benchmark was not built with Kokkos (see TNL_BENCHMARKS_USE_KOKKOS).\n";
   return false;
#endif
}

/**
 * \brief Runs the AMReX implementation of the given scheme, if AMReX supports the device in this build.
 */
template< int Dimension, bool Fvm, typename Real, typename Device >
bool
runAMReX( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
#ifdef TNL_BENCHMARKS_HAVE_AMREX
   if constexpr( TNL::Benchmarks::AMReXSupport::isSupported< Device >() ) {
      HeatEquationSolverBenchmarkAMReX< Dimension, Fvm, Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   else {
      std::cerr << "The AMReX implementation is not available for the device " << TNL::Benchmarks::getDeviceName< Device >()
                << " in this build, skipping it.\n";
      return true;
   }
#else
   std::cerr << "The benchmark was not built with AMReX (see TNL_BENCHMARKS_USE_AMREX).\n";
   return false;
#endif
}

/**
 * \brief Runs the GridTools implementation of the given scheme, if GridTools supports the device in this build.
 */
template< int Dimension, bool Fvm, typename Real, typename Device >
bool
runGridTools( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
#ifdef TNL_BENCHMARKS_HAVE_GRIDTOOLS
   if constexpr( TNL::Benchmarks::GridToolsSupport::isSupported< Device >() ) {
      HeatEquationSolverBenchmarkGridTools< Dimension, Fvm, Real, Device > solver;
      return solver.runBenchmark( benchmark, parameters );
   }
   else {
      std::cerr << "The GridTools implementation is not available for the device " << TNL::Benchmarks::getDeviceName< Device >()
                << " in this build, skipping it.\n";
      return true;
   }
#else
   std::cerr << "The benchmark was not built with GridTools (see TNL_BENCHMARKS_USE_GRIDTOOLS).\n";
   return false;
#endif
}

template< int Dimension, typename Real, typename Device >
bool
runImplementation( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto scheme = parameters.getParameter< TNL::String >( "scheme" );
   const auto implementation = parameters.getParameter< TNL::String >( "implementation" );
   if( scheme == "fdm" ) {
      if( implementation == "parallel-for" ) {
         HeatEquationSolverBenchmarkFDMParallelFor< Dimension, Real, Device > solver;
         return solver.runBenchmark( benchmark, parameters );
      }
      if( implementation == "nd-array" ) {
         HeatEquationSolverBenchmarkFDMNDArray< Dimension, Real, Device > solver;
         return solver.runBenchmark( benchmark, parameters );
      }
      if( implementation == "grid" ) {
         HeatEquationSolverBenchmarkFDMGrid< Dimension, Real, Device > solver;
         return solver.runBenchmark( benchmark, parameters );
      }
      if( implementation == "kokkos" )
         return runKokkos< Dimension, false, Real, Device >( benchmark, parameters );
      if( implementation == "amrex" )
         return runAMReX< Dimension, false, Real, Device >( benchmark, parameters );
      if( implementation == "gridtools" )
         return runGridTools< Dimension, false, Real, Device >( benchmark, parameters );
      std::cerr << "Unknown implementation " << implementation << " for " << scheme << " numerical scheme.\n";
      return false;
   }
   if( scheme == "fvm" ) {
      if( implementation == "parallel-for" ) {
         HeatEquationSolverBenchmarkFVMParallelFor< Dimension, Real, Device > solver;
         return solver.runBenchmark( benchmark, parameters );
      }
      if( implementation == "grid" ) {
         HeatEquationSolverBenchmarkFVMGrid< Dimension, Real, Device > solver;
         return solver.runBenchmark( benchmark, parameters );
      }
      if( implementation == "kokkos" )
         return runKokkos< Dimension, true, Real, Device >( benchmark, parameters );
      if( implementation == "amrex" )
         return runAMReX< Dimension, true, Real, Device >( benchmark, parameters );
      if( implementation == "gridtools" )
         return runGridTools< Dimension, true, Real, Device >( benchmark, parameters );
      if( implementation == "nd-array" )
         std::cerr << "\"nd-array\" implementation cannot be used for \"fvm\" numerical scheme.\n";
      else
         std::cerr << "Unknown implementation " << implementation << " for " << scheme << " numerical scheme.\n";
      return false;
   }
   std::cerr << "Unknown numerical scheme " << scheme << '\n';
   return false;
}

template< int Dimension, typename Real >
bool
resolveDevice( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto device = parameters.getParameter< TNL::String >( "device" );
   bool result = true;
   if( device == "sequential" || device == "all" )
      result = runImplementation< Dimension, Real, TNL::Devices::Sequential >( benchmark, parameters ) && result;
   if( device == "host" || device == "all" )
      result = runImplementation< Dimension, Real, TNL::Devices::Host >( benchmark, parameters ) && result;
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      result = runImplementation< Dimension, Real, TNL::Devices::GPU >( benchmark, parameters ) && result;
#endif
   return result;
}

template< typename Real >
bool
resolveDimension( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const int dimension = parameters.getParameter< int >( "dimension" );
   if( dimension == 1 )
      return resolveDevice< 1, Real >( benchmark, parameters );
   if( dimension == 2 )
      return resolveDevice< 2, Real >( benchmark, parameters );
   if( dimension == 3 )
      return resolveDevice< 3, Real >( benchmark, parameters );
   std::cerr << "Wrong dimension " << dimension << ", only 1D, 2D and 3D problems are allowed.\n";
   return false;
}

bool
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto precision = parameters.getParameter< TNL::String >( "precision" );
   bool result = true;
   if( precision == "all" || precision == "float" )
      result = resolveDimension< float >( benchmark, parameters ) && result;
   if( precision == "all" || precision == "double" )
      result = resolveDimension< double >( benchmark, parameters ) && result;
   return result;
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription config;

   configSetup( config );

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   // the libraries for the comparison must be initialized after the device setup
   TNL::Benchmarks::ComparisonLibrariesGuard comparisonLibraries( argv[ 0 ] );

   if( ! resolvePrecision( benchmark, parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

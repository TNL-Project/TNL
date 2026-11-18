// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

#include "MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid.h"
#include "MeanCurvatureFlowSemiImplicitSolverBenchmarkFDMGrid.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "Mean-curvature flow benchmark settings:" );
   config.addEntry< TNL::String >( "implementation", "Implementation of the mean-curvature flow solver.", "grid" );
   config.addEntryEnum< TNL::String >( "grid" );
   config.addEntry< TNL::String >( "scheme", "Numerical scheme used for the discretisation.", "fdm" );
   config.addEntryEnum< TNL::String >( "fdm" );
   config.addEntry< TNL::String >( "time-discretisation", "Type of discretisation in time.", "semi-implicit" );
   config.addEntryEnum< TNL::String >( "explicit" );
   config.addEntryEnum< TNL::String >( "semi-implicit" );

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

   MeanCurvatureFlowSolverBenchmark<>::configSetup( config );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
}

template< int Dimension, typename Real, typename Device >
bool
runImplementation( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto scheme = parameters.getParameter< TNL::String >( "scheme" );
   const auto implementation = parameters.getParameter< TNL::String >( "implementation" );
   const auto timeDiscretisation = parameters.getParameter< TNL::String >( "time-discretisation" );
   if( scheme == "fdm" ) {
      if( implementation == "grid" ) {
         if( timeDiscretisation == "explicit" ) {
            MeanCurvatureFlowExplicitSolverBenchmarkFDMGrid< Dimension, Real, Device > solver;
            return solver.runBenchmark( benchmark, parameters );
         }
         MeanCurvatureFlowSemiImplicitSolverBenchmarkFDMGrid< Dimension, Real, Device > solver;
         return solver.runBenchmark( benchmark, parameters );
      }
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

   if( ! resolvePrecision( benchmark, parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

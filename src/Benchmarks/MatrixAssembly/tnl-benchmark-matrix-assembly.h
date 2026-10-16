// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <exception>
#include <iostream>
#include <memory>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/MPI/Config.h>
#include <TNL/MPI/ScopedInitializer.h>

#include "MatrixAssemblyBenchmark.h"

#ifdef HAVE_PETSC
//! \brief RAII wrapper for the initialization and finalization of PETSc.
struct PetscScope
{
   PetscScope()
   {
      TNL::Benchmarks::MatrixAssembly::checkPetsc( PetscInitializeNoArguments(), "PetscInitializeNoArguments" );
   }

   PetscScope( const PetscScope& ) = delete;
   PetscScope&
   operator=( const PetscScope& ) = delete;

   ~PetscScope()
   {
      PetscFinalize();
   }
};
#endif

void
configSetup( TNL::Config::ConfigDescription& config )
{
   TNL::Benchmarks::Benchmark::configSetup( config );

   config.addDelimiter( "Matrix assembly benchmark settings:" );
   config.addEntry< std::string >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "device", "Device to run benchmarks on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );
   config.addEntry< std::string >( "dimension", "Dimension of the numerical grid.", "all" );
   config.addEntryEnum( "1" );
   config.addEntryEnum( "2" );
   config.addEntryEnum( "3" );
   config.addEntryEnum( "all" );
   config.addEntry< int >(
      "x-size", "Number of grid nodes along the x axis (0 means 10^6, 10^3 and 10^2 nodes in 1D, 2D and 3D).", 0 );
   config.addEntry< int >( "y-size", "Number of grid nodes along the y axis (0 means 10^3 and 10^2 nodes in 2D and 3D).", 0 );
   config.addEntry< int >( "z-size", "Number of grid nodes along the z axis (0 means 10^2 nodes in 3D).", 0 );
   config.addList< std::string >(
      "libraries", "Libraries to benchmark (TNL, Ginkgo, Hypre, PETSc, cuSPARSE, all).", { "all" } );
   config.addList< std::string >( "formats", "Matrix formats to benchmark (CSR, Ellpack, SlicedEllpack, all).", { "all" } );
   config.addEntry< bool >( "verify", "Verify the assembled matrices.", true );

   config.addDelimiter( "Device settings:" );
   TNL::Devices::Host::configSetup( config );
   TNL::Devices::GPU::configSetup( config );
   TNL::MPI::configSetup( config );
}

void
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters )
{
   const auto& precision = parameters.getParameter< std::string >( "precision" );

   if( precision == "all" || precision == "float" )
      TNL::Benchmarks::MatrixAssembly::runBenchmark< float, int >( benchmark, parameters );
   if( precision == "all" || precision == "double" )
      TNL::Benchmarks::MatrixAssembly::runBenchmark< double, int >( benchmark, parameters );
}

int
main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi( argc, argv );

   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   try {
#ifdef HAVE_HYPRE
      TNL::Hypre hypre;
#endif
#ifdef HAVE_PETSC
      PetscScope petsc;
#endif

      // init benchmark
      TNL::Benchmarks::Benchmark benchmark;
      benchmark.setup( parameters, argv[ 0 ] );

      resolvePrecision( benchmark, parameters );
   }
   catch( const std::exception& e ) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

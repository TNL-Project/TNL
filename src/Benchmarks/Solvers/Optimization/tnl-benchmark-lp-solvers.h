// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <set>
#include <sstream>
#include <string>

#ifndef NDEBUG
   #include <TNL/Debugging/FPE.h>
#endif

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Solvers/Optimization/PDLP.h>
#include <TNL/Solvers/Optimization/LPProblem.h>
#include <TNL/Solvers/Optimization/LPProblemReader.h>

#include <TNL/Benchmarks/Benchmarks.h>

#include "GurobiLPBenchmark.h"
#include "ORToolsLPBenchmark.h"

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Value, typename Index >
void
benchmarkLPSolvers( Benchmark<>& benchmark, const Config::ParameterContainer& parameters, size_t dofs )
{}

template< typename Value, typename Device, typename Index >
struct LPSolversBenchmark
{
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = Index;
   using SolverMonitorType = typename Benchmark<>::SolverMonitorType;

   template< typename SolverType >
   static bool
   benchmarkSolver( Benchmark<>& benchmark, const Config::ParameterContainer& parameters, const char* solverName )
   {
      using ValueType = typename SolverType::ValueType;
      using VectorType = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;
      //using VectorView = typename VectorType::ViewType;
      using MatrixType = TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType >;
      //using LPProblemType = TNL::Solvers::Optimization::LPProblem< MatrixType >;

      std::string device = "host";
      if( std::is_same_v< DeviceType, Devices::Cuda > ) {
         device = "cuda";
      }

      return true;
   }

   static bool
   run( Benchmark<>& benchmark, const Config::ParameterContainer& parameters )
   {
      using MatrixType = TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType >;
      using LPProblemType = TNL::Solvers::Optimization::LPProblem< MatrixType >;

      const String& fileName = parameters.getParameter< String >( "input-file" );

      // Gurobi solver
      if( parameters.getParameter< bool >( "with-gurobi" ) ) {
         std::cout << "Running Gurobi solver..." << std::endl;
         try {
            gurobiBenchmark( benchmark, fileName );
         }
         catch( GRBException& e ) {
            std::cerr << "Gurobi error: " << e.getMessage() << std::endl;
         }
         catch( ... ) {
            std::cerr << "An unexpected error occurred." << std::endl;
         }
      }

      std::cout << "Reading LP problem from file " << fileName << std::endl;
      TNL::Solvers::Optimization::LPProblemReader< LPProblemType > reader;
      auto lpProblem = reader.read( fileName );
      //std::cout << lpProblem << std::endl;

      if( parameters.getParameter< bool >( "with-ortools" ) ) {
         std::cout << "Running OR-Tools solver..." << std::endl;
         try {
            orToolsLPBenchmark( benchmark, lpProblem );
         }
         catch( ... ) {
            std::cerr << "An unexpected error occurred." << std::endl;
         }
      }

      if( parameters.getParameter< bool >( "with-tnl" ) ) {
         std::cout << "Running TNL PDLP solver..." << std::endl;
         typename LPProblemType::VectorType x( lpProblem.getVariableCount() );
         TNL::Solvers::Optimization::PDLP< LPProblemType > solver;
         auto [ converged, cost, error ] = solver.solve( lpProblem, x );
         std::cout << "Solution: " << x << std::endl;
      }
      return true;
   }
};

template< typename Real, typename Device >
bool
resolveIndexType( Benchmark<>& benchmark, Config::ParameterContainer& parameters )
{
   const String& index = parameters.getParameter< String >( "index-type" );
   if( index == "int" && ! LPSolversBenchmark< Real, Device, int >::run( benchmark, parameters ) )
      return false;
   if( index == "long int" && ! LPSolversBenchmark< Real, Device, long int >::run( benchmark, parameters ) )
      return false;
   return true;
}

template< typename Real >
bool
resolveDeviceType( Benchmark<>& benchmark, Config::ParameterContainer& parameters )
{
   const String& device = parameters.getParameter< String >( "device" );
   if( ( device == "sequential" || device == "all" )
       && ! resolveIndexType< Real, Devices::Sequential >( benchmark, parameters ) )
      return false;
   if( ( device == "host" || device == "all" ) && ! resolveIndexType< Real, Devices::Host >( benchmark, parameters ) )
      return false;
   if( device == "cuda" || device == "all" ) {
#ifdef __CUDACC__
      if( ! resolveIndexType< Real, Devices::Cuda >( benchmark, parameters ) )
         return false;
#else
      std::cerr << "CUDA support not compiled in." << std::endl;
      return false;
#endif
   }
   return true;
}

bool
resolveRealTypes( Benchmark<>& benchmark, Config::ParameterContainer& parameters )
{
   const String& realType = parameters.getParameter< String >( "precision" );
   if( ( realType == "float" || realType == "all" ) && ! resolveDeviceType< float >( benchmark, parameters ) )
      return false;
   if( ( realType == "double" || realType == "all" ) && ! resolveDeviceType< double >( benchmark, parameters ) )
      return false;
   return true;
}

void
configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "input-file", "Input file name." );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-lp-solvers.log" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addList< String >( "solvers", "List of solvers to run benchmarks for.", { "all" } );
   config.addEntry< String >( "device", "Run benchmarks using given device.", "host" );
   config.addEntryEnum( "sequential" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "all" );
   config.addEntry< bool >( "with-gurobi", "Run benchmarks with Gurobi solver.", false );
   config.addEntry< bool >( "with-ortools", "Run benchmarks with OR-Tools solver.", false );
   config.addEntry< bool >( "with-tnl", "Run benchmarks with TNL PDLP solver.", true );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< String >( "index-type", "Run benchmarks with given index type.", "int" );
   config.addEntryEnum< String >( "int" );
   config.addEntryEnum< String >( "long int" );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "LP solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   //using Vector = TNL::Containers::Vector< int >;
}

int
main( int argc, char* argv[] )
{
#ifndef NDEBUG
   Debugging::trackFloatingPointExceptions();
#endif

   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   //TNL::MPI::ScopedInitializer mpi( argc, argv );
   //const int rank = TNL::MPI::GetRank();

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! Devices::Host::setup( parameters ) || ! Devices::Cuda::setup( parameters ) )  // || ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const String& logFileName = parameters.getParameter< String >( "log-file" );
   const String& outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   //const int verbose = ( rank == 0 ) ? parameters.getParameter< int >( "verbose" ) : 0;
   const int verbose = parameters.getParameter< int >( "verbose" );

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream logFile;
   //if( rank == 0 )
   logFile.open( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   std::cout << "Running benchmarks for LP solvers." << std::endl;
   return ! resolveRealTypes( benchmark, parameters );
}

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
#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Solvers/ODE/Merson.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include "ODESolversBenchmarkResult.h"
#include "Euler.h"
#include "Merson.h"

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Pointers;

template< typename Real, typename Index >
void
benchmarkODESolvers( Benchmark<>& benchmark, const Config::ParameterContainer& parameters, size_t dofs )
{
}

template< typename Real, typename Device, typename Index >
struct ODESolversBenchmark
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using VectorView = typename VectorType::ViewType;
   using SolverMonitorType = typename Benchmark<>::SolverMonitorType;

   template< typename SolverType >
   static bool
   benchmarkSolver( Benchmark<>& benchmark,
                    const Config::ParameterContainer& parameters,
                    const char* solverName )
   {
      std::string device = "host";
      if( std::is_same< DeviceType, Devices::Cuda >::value ) {
         device = "cuda";
      }

      SolverType solver;
      if constexpr( std::is_same< SolverType, TNL::Solvers::ODE::Merson< VectorType, SolverMonitorType > >::value ||
                    std::is_same< SolverType, TNL::Benchmarks::Merson< VectorType, SolverMonitorType > >::value ) {
         solver.setAdaptivity( 0.0 );
      }
      RealType tau = 0.1;
      std::size_t dofs = parameters.getParameter< int >( "size" );
      VectorType u( dofs, 0.0 );
      ODESolversBenchmarkResult< RealType, DeviceType, IndexType > benchmarkResult( 1.0, u );
      for( int eoc_steps = 0; eoc_steps < 5; eoc_steps++ ) {
         benchmark.setMetadataColumns(
            TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< RealType >() },
                                                             { "index type", TNL::getType< IndexType >() },
                                                             { "solver", std::string( solverName ) },
                                                             { "DOFs", convertToString( dofs ) }
                                                           } ) );
         solver.setTime( 0.0 );
         solver.setTau( tau );
         solver.setStopTime( 1.0 );
         u = 0;
         std::size_t iterations = 1.0 / tau + 1;
         benchmark.setDatasetSize( dofs * sizeof( RealType ) * iterations );
         auto problem = [=] ( const RealType& t, const RealType& tau, const VectorView& u_view, VectorView& fu_view ) {
            auto computeF = [=] __cuda_callable__ ( IndexType i ) mutable {
               fu_view[ i ] = 6.0 * TNL::pow( t, 5.0 );
            };
            Algorithms::parallelFor< DeviceType >( 0, u_view.getSize(), computeF );
         };
         auto solve = [&]() { solver.solve( u, problem ); };
         benchmark.time< Devices::Host >( device, solve, benchmarkResult );
         tau /= 2.0;
      }
      return true;
   }

   static bool
   run( Benchmark<>& benchmark, const Config::ParameterContainer& parameters )
   {
      const auto& solvers = parameters.getList< String >( "solvers" );
      for( auto&& solver : solvers )
      {
         if( solver == "euler" || solver == "all" ) {
            using Solver = Solvers::ODE::Euler< VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Euler" );
            using SolverNonET = Benchmarks::Euler< VectorType, SolverMonitorType >;
            benchmarkSolver< SolverNonET >( benchmark, parameters, "Euler non-ET" );
         }
         if( solver == "merson" || solver == "all" ) {
            using Solver = Solvers::ODE::Merson< VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Merson" );
            using SolverNonET = Benchmarks::Merson< VectorType, SolverMonitorType >;
            benchmarkSolver< SolverNonET >( benchmark, parameters, "Merson non-ET");
         }
      }
      return true;
   }
};

template< typename Real, typename Device >
bool resolveIndexType( Benchmark<>& benchmark,
                       Config::ParameterContainer& parameters )
{
   const String& index = parameters.getParameter< String >( "index-type" );
   if( index == "int" && ! ODESolversBenchmark< Real, Device, int >::run( benchmark, parameters ) )
      return false;
   if( index == "long int" && ! ODESolversBenchmark< Real, Device, long int >::run( benchmark, parameters ) )
      return false;
   return true;
}

template< typename Real >
bool resolveDeviceType( Benchmark<>& benchmark,
                        Config::ParameterContainer& parameters )
{
   const String& device = parameters.getParameter< String >( "device" );
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
   if( ( realType == "float" || realType == "all" ) &&
       ! resolveDeviceType< float >( benchmark, parameters ) )
      return false;
   if( ( realType == "double" || realType == "all" ) &&
       ! resolveDeviceType< double >( benchmark, parameters ) )
      return false;
   return true;
}

void
configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-ode-solvers.log" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addList< String >( "solvers", "List of solvers to run benchmarks for.", { "all" } );
   config.addEntryEnum< String >( "euler" );
   config.addEntryEnum< String >( "merson" );
   config.addEntryEnum< String >( "all" );
   config.addEntry< String >( "device", "Run benchmarks using given device.", "host" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "all" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< String >( "index-type", "Run benchmarks with given index type.", "int" );
   config.addEntryEnum< String >( "int" );
   config.addEntryEnum< String >( "long int" );
   config.addEntry< int >( "size", "Size of the ODE system (all ODEs are the same).", 1<<20 );
   config.addEntry< double >( "final-time", "Final time of the benchmark test.", 1.0 );
   config.addEntry< double >( "time-step", "Time step of the benchmark test.", 1.0e-2 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "ODE solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   using Vector = TNL::Containers::Vector< int>;
   Solvers::ODE::Euler< Vector >::configSetup( config );
   Solvers::ODE::Merson< Vector >::configSetup( config );
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

   TNL::MPI::ScopedInitializer mpi( argc, argv );
   const int rank = TNL::MPI::GetRank();

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! Devices::Host::setup( parameters ) || ! Devices::Cuda::setup( parameters ) || ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const String& logFileName = parameters.getParameter< String >( "log-file" );
   const String& outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = ( rank == 0 ) ? parameters.getParameter< int >( "verbose" ) : 0;

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream logFile;
   if( rank == 0 )
      logFile.open( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   return ! resolveRealTypes( benchmark, parameters );
}

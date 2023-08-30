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
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>
#include <TNL/Solvers/ODE/Methods/Merson.h>
#include <TNL/Solvers/ODE/Methods/DormandPrince.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include "ODESolversBenchmarkResult.h"
#include "Legacy/EulerNonET.h"
#include "Legacy/MersonNonET.h"
#include "Legacy/Euler.h"
#include "Legacy/Merson.h"


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
   using SolverMonitorType = typename Benchmark<>::SolverMonitorType;

   template< typename SolverType >
   static bool
   benchmarkSolver( Benchmark<>& benchmark,
                    const Config::ParameterContainer& parameters,
                    const char* solverName )
   {
      using ValueType = typename SolverType::ValueType;
      using VectorType = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;
      using VectorView = typename VectorType::ViewType;

      std::string device = "host";
      if( std::is_same< DeviceType, Devices::Cuda >::value ) {
         device = "cuda";
      }

      double adaptivity = parameters.getParameter< double >( "adaptivity" );
      SolverType solver;
      if constexpr( ! std::is_same< SolverType, Euler< VectorType, SolverMonitorType > >::value &&
                    ! std::is_same< SolverType, TNL::Benchmarks::EulerNonET< VectorType, SolverMonitorType > >::value ) {
         solver.setAdaptivity( adaptivity );
      }

      RealType tau = 0.5;
      std::size_t dofs = parameters.getParameter< int >( "size" ) * sizeof( RealType ) / sizeof( ValueType );
      VectorType u( dofs, 0.0 );
      RealType correct_solution = exp( 1.0 ) - exp( 0.0 );
      ODESolversBenchmarkResult< SolverType, DeviceType, IndexType > benchmarkResult( correct_solution, solver, u );
      int eoc_steps_count = adaptivity ? 1 : 5;
      for( int eoc_steps = 0; eoc_steps < eoc_steps_count; eoc_steps++ ) {
         benchmark.setMetadataColumns(
            TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< RealType >() },
                                                             { "index type", TNL::getType< IndexType >() },
                                                             { "solver", std::string( solverName ) },
                                                             { "DOFs", convertToString( dofs ) }
                                                           } ) );
         solver.setStopTime( 1.0 );
         u = 0;
         std::size_t iterations = 1.0 / tau + 1;
         benchmark.setDatasetSize( dofs * sizeof( ValueType ) * iterations );
         auto reset_u = [&] () mutable { u = 0.0; };
         if constexpr( SolverType::isStatic() ) {
            auto u_view = u.getView();
            auto solve = [&]() {
               auto problem = [] __cuda_callable__ ( const RealType& t, const RealType& tau, const ValueType& u, ValueType& fu ) {
                     fu = TNL::exp( t );
               };
               TNL::Algorithms::parallelFor< DeviceType >( 0, u.getSize(), [=] __cuda_callable__ ( IndexType i ) mutable {
                  solver.setTime( 0.0 );
                  solver.setTau( tau );
                  solver.setAdaptivity( adaptivity );
                  solver.solve( u_view[ i ], problem );
               } );
            };
            benchmark.time< DeviceType >( reset_u, device, solve, benchmarkResult );
         } else {
            auto problem = [=] ( const RealType& t, const RealType& tau, const VectorView& u_view, VectorView& fu_view ) {
               auto computeF = [=] __cuda_callable__ ( IndexType i ) mutable {
                  fu_view[ i ] = TNL::exp( t );
               };
               Algorithms::parallelFor< DeviceType >( 0, u_view.getSize(), computeF );
            };
            auto solve = [&]() {
               solver.setStopTime( 1.0 );
               solver.setTime( 0.0 );
               solver.setTau( tau );
               solver.solve( u, problem );
            };
            benchmark.time< DeviceType >( reset_u, device, solve, benchmarkResult );
         }
         tau /= 2.0;
      }
      return true;
   }

   static bool
   run( Benchmark<>& benchmark, const Config::ParameterContainer& parameters )
   {
      using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
      const auto& solvers = parameters.getList< String >( "solvers" );
      for( auto&& solver : solvers )
      {
         if( solver == "euler" || solver == "all" ) {
            using LegacySolverNonET = Benchmarks::EulerNonET< VectorType, SolverMonitorType >;
            benchmarkSolver< LegacySolverNonET >( benchmark, parameters, "Leg. Euler non-ET" );
            using LegacySolver = Euler< VectorType, SolverMonitorType >;
            benchmarkSolver< LegacySolver >( benchmark, parameters, "Leg. Euler" );
            using Method = TNL::Solvers::ODE::Methods::Euler< RealType >;
            using VectorSolver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< VectorSolver >( benchmark, parameters, "Euler" );
            using StaticSolver_1 = TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 1, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_1 >( benchmark, parameters, "Euler SV-1" );
            using StaticSolver_2 = TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 2, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_2 >( benchmark, parameters, "Euler SV-2" );
         }
         if( solver == "merson" || solver == "all" ) {
            using LegacySolverNonET = Benchmarks::MersonNonET< VectorType, SolverMonitorType >;
            benchmarkSolver< LegacySolverNonET >( benchmark, parameters, "Leg. Merson non-ET");
            using LegacySolver = Merson< VectorType, SolverMonitorType >;
            benchmarkSolver< LegacySolver >( benchmark, parameters, "Leg. Merson" );
            using Method = TNL::Solvers::ODE::Methods::Merson< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Merson" );
            using StaticSolver_1 = TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 1, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_1 >( benchmark, parameters, "Merson SV-1" );
            using StaticSolver_2 = TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 2, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_2 >( benchmark, parameters, "Merson SV-2" );
         }
         if( solver == "dormand-prince" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::DormandPrince< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Dormand-Prince" );
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
   if( ( device == "sequential" || device == "all" ) && ! resolveIndexType< Real, Devices::Sequential >( benchmark, parameters ) )
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
   config.addEntryEnum( "sequential" );
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
   config.addEntry< double >( "adaptivity", "Set adaptive time stepping. Zero means no adaptive time stepping", 0.0 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "ODE solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   using Vector = TNL::Containers::Vector< int>;
   Euler< Vector >::configSetup( config );
   Merson< Vector >::configSetup( config );
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

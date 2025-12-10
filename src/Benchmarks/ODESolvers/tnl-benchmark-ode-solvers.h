// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

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
#include <TNL/Solvers/ODE/Methods/BogackiShampin.h>
#include <TNL/Solvers/ODE/Methods/CashKarp.h>
#include <TNL/Solvers/ODE/Methods/DormandPrince.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg2.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg5.h>
#include <TNL/Solvers/ODE/Methods/Heun2.h>
#include <TNL/Solvers/ODE/Methods/Heun3.h>
#include <TNL/Solvers/ODE/Methods/Kutta.h>
#include <TNL/Solvers/ODE/Methods/KuttaMerson.h>
#include <TNL/Solvers/ODE/Methods/Midpoint.h>
#include <TNL/Solvers/ODE/Methods/OriginalRungeKutta.h>
#include <TNL/Solvers/ODE/Methods/Ralston2.h>
#include <TNL/Solvers/ODE/Methods/Ralston3.h>
#include <TNL/Solvers/ODE/Methods/Ralston4.h>
#include <TNL/Solvers/ODE/Methods/Rule38.h>
#include <TNL/Solvers/ODE/Methods/SSPRK3.h>
#include <TNL/Solvers/ODE/Methods/VanDerHouwenWray.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include "ODESolversBenchmarkResult.h"
#include "Legacy/EulerNonET.h"
#include "Legacy/MersonNonET.h"
#include "Legacy/Euler.h"
#include "Legacy/Merson.h"

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Real, typename Index >
void
benchmarkODESolvers( Benchmark<>& benchmark, const Config::ParameterContainer& parameters, size_t dofs )
{}

template< typename Real, typename Device, typename Index >
struct ODESolversBenchmark
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using SolverMonitorType = typename Benchmark<>::SolverMonitorType;

   template< typename SolverType >
   static bool
   benchmarkSolver( Benchmark<>& benchmark, const Config::ParameterContainer& parameters, const char* solverName )
   {
      using ValueType = typename SolverType::ValueType;
      using ElementType = typename SolverElementType< SolverType >::type;
      using VectorType = TNL::Containers::Vector< ElementType, DeviceType, IndexType >;
      using VectorView = typename VectorType::ViewType;

      std::string device = "host";
      if( std::is_same_v< DeviceType, Devices::Cuda > ) {
         device = "cuda";
      }

      double adaptivity = parameters.getParameter< double >( "adaptivity" );
      SolverType solver;
      if constexpr( ! std::is_same_v< SolverType, Euler< VectorType, SolverMonitorType > >
                    && ! std::is_same_v< SolverType, TNL::Benchmarks::EulerNonET< VectorType, SolverMonitorType > > )
         solver.setAdaptivity( adaptivity );

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
                                                             { "DOFs", convertToString( dofs ) } } ) );
         solver.setStopTime( 1.0 );
         u = 0;
         std::size_t iterations = 1.0 / tau + 1;
         benchmark.setDatasetSize( dofs * sizeof( ValueType ) * iterations );
         auto reset_u = [ & ]() mutable
         {
            u = 0.0;
         };
         if constexpr( SolverType::isStatic() ) {
            auto u_view = u.getView();
            auto solve = [ & ]()
            {
               auto problem =
                  [] __cuda_callable__( const RealType& t, const RealType& tau, const ElementType& u, ElementType& fu )
               {
                  fu = TNL::exp( t );
               };
               TNL::Algorithms::parallelFor< DeviceType >( 0,
                                                           u.getSize(),
                                                           [ = ] __cuda_callable__( IndexType i ) mutable
                                                           {
                                                              solver.setTime( 0.0 );
                                                              solver.setTau( tau );
                                                              solver.setAdaptivity( adaptivity );
                                                              solver.solve( u_view[ i ], problem );
                                                           } );
            };
            benchmark.time< DeviceType >( reset_u, device, solve, benchmarkResult );
         }
         else {
            auto problem = [ = ]( const RealType& t, const RealType& tau, const VectorView& u_view, VectorView& fu_view )
            {
               auto computeF = [ = ] __cuda_callable__( IndexType i ) mutable
               {
                  fu_view[ i ] = TNL::exp( t );
               };
               Algorithms::parallelFor< DeviceType >( 0, u_view.getSize(), computeF );
            };
            auto solve = [ & ]()
            {
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
      const bool legacy_solvers = parameters.getParameter< bool >( "legacy-solvers" );
      for( auto&& solver : solvers ) {
         if( solver == "euler" || solver == "all" ) {
            if( legacy_solvers ) {
               using LegacySolverNonET = Benchmarks::EulerNonET< VectorType, SolverMonitorType >;
               benchmarkSolver< LegacySolverNonET >( benchmark, parameters, "Leg. Euler non-ET" );
               using LegacySolver = Euler< VectorType, SolverMonitorType >;
               benchmarkSolver< LegacySolver >( benchmark, parameters, "Leg. Euler" );
            }
            using Method = TNL::Solvers::ODE::Methods::Euler< RealType >;
            using VectorSolver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< VectorSolver >( benchmark, parameters, "Euler" );
            using StaticSolver_1 =
               TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 1, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_1 >( benchmark, parameters, "Euler SV-1" );
            using StaticSolver_2 =
               TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 2, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_2 >( benchmark, parameters, "Euler SV-2" );
         }
         if( solver == "kutta-merson" || solver == "all" ) {
            if( legacy_solvers ) {
               using LegacySolverNonET = Benchmarks::MersonNonET< VectorType, SolverMonitorType >;
               benchmarkSolver< LegacySolverNonET >( benchmark, parameters, "Leg. Kutta-Merson non-ET" );
               using LegacySolver = Merson< VectorType, SolverMonitorType >;
               benchmarkSolver< LegacySolver >( benchmark, parameters, "Leg. Kutta-Merson" );
            }
            using Method = TNL::Solvers::ODE::Methods::KuttaMerson< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Kutta-Merson" );
            using StaticSolver_1 =
               TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 1, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_1 >( benchmark, parameters, "Kutta-Merson SV-1" );
            using StaticSolver_2 =
               TNL::Solvers::ODE::ODESolver< Method, Containers::StaticVector< 2, Real >, SolverMonitorType >;
            benchmarkSolver< StaticSolver_2 >( benchmark, parameters, "Kutta-Merson SV-2" );
         }
         if( solver == "bogacki-shampin" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::BogackiShampin< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Bogacki-Shampin" );
         }
         if( solver == "dormand-prince" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::DormandPrince< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Dormand-Prince" );
         }
         if( solver == "fehlberg2" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Fehlberg2< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Fehlberg2" );
         }
         if( solver == "fehlberg5" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Fehlberg5< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Fehlberg5" );
         }
         if( solver == "heun2" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Heun2< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Heun2" );
         }
         if( solver == "heun3" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Heun3< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Heun3" );
         }
         if( solver == "kutta" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Kutta< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Kutta" );
         }
         if( solver == "ralston2" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Ralston2< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Ralston2" );
         }
         if( solver == "ralston3" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Ralston2< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Ralston3" );
         }
         if( solver == "ralston4" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Ralston4< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Ralston4" );
         }
         if( solver == "original-runge-kutta" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::OriginalRungeKutta< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Runge-Kutta" );
         }
         if( solver == "vanderhouwen-wray" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::VanDerHouwenWray< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "VanDerHouwen-Wray" );
         }
         if( solver == "cash-karp" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::CashKarp< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Cash-Karp" );
         }
         if( solver == "midpoint" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Midpoint< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Midpoint" );
         }
         if( solver == "rule38" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::Rule38< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "Rule38" );
         }
         if( solver == "ssprk3" || solver == "all" ) {
            using Method = TNL::Solvers::ODE::Methods::SSPRK3< RealType >;
            using Solver = TNL::Solvers::ODE::ODESolver< Method, VectorType, SolverMonitorType >;
            benchmarkSolver< Solver >( benchmark, parameters, "SSPRK3" );
         }
      }
      return true;
   }
};

template< typename Real, typename Device >
bool
resolveIndexType( Benchmark<>& benchmark, Config::ParameterContainer& parameters )
{
   const String& index = parameters.getParameter< String >( "index-type" );
   if( index == "int" && ! ODESolversBenchmark< Real, Device, int >::run( benchmark, parameters ) )
      return false;
   if( index == "long int" && ! ODESolversBenchmark< Real, Device, long int >::run( benchmark, parameters ) )
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
      std::cerr << "CUDA support not compiled in.\n";
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
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-ode-solvers.log" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addList< String >( "solvers", "List of solvers to run benchmarks for.", { "all" } );
   config.addEntryEnum< String >( "bogacki-shampin" );
   config.addEntryEnum< String >( "cash-karp" );
   config.addEntryEnum< String >( "dormand-prince" );
   config.addEntryEnum< String >( "euler" );
   config.addEntryEnum< String >( "fehlberg2" );
   config.addEntryEnum< String >( "fehlberg5" );
   config.addEntryEnum< String >( "heun2" );
   config.addEntryEnum< String >( "heun3" );
   config.addEntryEnum< String >( "kutta" );
   config.addEntryEnum< String >( "kutta-merson" );
   config.addEntryEnum< String >( "midpoint" );
   config.addEntryEnum< String >( "ralston2" );
   config.addEntryEnum< String >( "ralston3" );
   config.addEntryEnum< String >( "ralston4" );
   config.addEntryEnum< String >( "rule38" );
   config.addEntryEnum< String >( "original-runge-kutta" );
   config.addEntryEnum< String >( "ssprk3" );
   config.addEntryEnum< String >( "vanderhouwen-wray" );
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
   config.addEntry< int >( "size", "Size of the ODE system (all ODEs are the same).", 1 << 20 );
   config.addEntry< double >( "final-time", "Final time of the benchmark test.", 1.0 );
   config.addEntry< double >( "time-step", "Time step of the benchmark test.", 1.0e-2 );
   config.addEntry< double >( "adaptivity", "Set adaptive time stepping. Zero means no adaptive time stepping", 0.0 );
   config.addEntry< bool >( "legacy-solvers", "Run benchmarks even for legacy implementations of the solvers.", false );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "ODE solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   using Vector = TNL::Containers::Vector< int >;
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

   const bool status = resolveRealTypes( benchmark, parameters );
   return static_cast< int >( ! status );
}

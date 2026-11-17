// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Utils.h>
#include <TNL/Benchmarks/JsonLogging.h>
#include "GraphBenchmarkBFS.h"
#include "GraphList.h"

void
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Precision settings:" );
   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
}

template< typename Real >
bool
startBenchmark( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   TNL::Benchmarks::Graphs::GraphBenchmarkProgress progress(
      TNL::Benchmarks::Graphs::benchmarkProgressDir( parameters, "bfs" ),
      TNL::Benchmarks::Graphs::progressGraphName( parameters ),
      TNL::Benchmarks::Graphs::mpiRank(),
      TNL::Benchmarks::Graphs::mpiSize() );

   // The GAP graphs with more than 2^31 stored edges need a 64-bit index, the other graphs use a 32-bit index.
   if( TNL::Benchmarks::Graphs::GraphBenchmarkBFS< Real, int >::requiresLongIndex( parameters ) ) {
      TNL::Benchmarks::Graphs::GraphBenchmarkBFS< Real, std::int64_t > graphBenchmark( parameters, progress );
      return graphBenchmark.runBenchmark( benchmark );
   }
   TNL::Benchmarks::Graphs::GraphBenchmarkBFS< Real, int > graphBenchmark( parameters, progress );
   return graphBenchmark.runBenchmark( benchmark );
}

bool
resolvePrecision( TNL::Benchmarks::Benchmark& benchmark, TNL::Config::ParameterContainer& parameters )
{
   // BFS never uses edge weights, only connectivity, so it always benchmarks
   // a binary (bool) adjacency matrix regardless of --precision. The option
   // is still accepted (and ignored) so that run-tnl-benchmark-graphs, which
   // passes --precision unconditionally to every benchmark binary, keeps
   // working unmodified.
   return startBenchmark< bool >( benchmark, parameters );
}

int
main( int argc, char* argv[] )
{
   // Lets several MPI ranks split one graph's GPU kernel sweep between them, each bound to its
   // own GPU (see the TNL::MPI::selectGPU() call below). MPI is initialized only when the process
   // was actually launched via `mpirun -np N` (or `srun -n N` with an MPI plugin), see
   // startedAsSeveralMPIProcesses(). HAVE_MPI is only defined when MPI was found by CMake.
#ifdef HAVE_MPI
   std::optional< TNL::MPI::ScopedInitializer > mpi;
   if( TNL::Benchmarks::Graphs::startedAsSeveralMPIProcesses() )
      mpi.emplace( argc, argv );
#endif

   TNL::Config::ConfigDescription config;
   TNL::Benchmarks::Graphs::GraphBenchmarkBFS<>::configSetup( config );
   configSetup( config );

   TNL::Config::ParameterContainer parameters;

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! TNL::Devices::Host::setup( parameters ) || ! TNL::Devices::GPU::setup( parameters ) )
      return EXIT_FAILURE;
#ifdef HAVE_MPI
   // Overrides whatever --cuda-device/--hip-device set (including the default) with a device
   // chosen from this rank's position on the node - only when actually running with more than
   // one rank, so a single-process run (the common case) keeps today's --cuda-device-driven
   // behavior untouched.
   if( TNL::Benchmarks::Graphs::mpiSize() > 1 )
      TNL::MPI::selectGPU();
#endif

   // init benchmark
   TNL::Benchmarks::Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   // Benchmark::setup() only wires up logging for rank 0 ("only root rank initializes loggers",
   // TNL's convention for the usual case where every rank computes the same collective result and
   // only rank 0's report matters). Here every rank benchmarks a different, equally significant
   // slice of kernels, so every other rank needs its own JSONL log - appended with a rank suffix
   // so it never collides with rank 0's own --log-file. rankLogFile must outlive `benchmark`.
   std::ofstream rankLogFile;
#ifdef HAVE_MPI
   if( TNL::Benchmarks::Graphs::mpiRank() > 0 ) {
      TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      if( logFileName.empty() )
         logFileName = std::filesystem::path( argv[ 0 ] ).stem().string() + ".log";
      const std::string rankLogFileName =
         std::string( logFileName.getString() ) + ".rank" + std::to_string( TNL::Benchmarks::Graphs::mpiRank() );
      const TNL::String& outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      rankLogFile.open( rankLogFileName, mode );
      benchmark.addLogger(
         std::make_unique< TNL::Benchmarks::JsonLogging >( rankLogFile, parameters.getParameter< int >( "verbose" ) ) );
   }
#endif

   const TNL::String& graphListFile = parameters.getParameter< TNL::String >( "graph-list" );
   bool ok = true;
   if( ! graphListFile.empty() ) {
      const std::string basePath = parameters.getParameter< TNL::String >( "graph-list-base-path" ).getString();
      const std::string progressDir = TNL::Benchmarks::Graphs::benchmarkProgressDir( parameters, "bfs" );
      TNL::Benchmarks::Graphs::runOverGraphList(
         parameters,
         graphListFile.getString(),
         basePath,
         progressDir,
         [ & ]()
         {
            ok = resolvePrecision( benchmark, parameters ) && ok;
         } );
   }
   else {
      if( parameters.getParameter< TNL::String >( "input-file" ).empty() ) {
         std::cerr << "ERROR: Input file name is required (or pass --graph-list)." << std::endl;
         return EXIT_FAILURE;
      }
      TNL::Benchmarks::Graphs::runSingleGraph(
         parameters,
         TNL::Benchmarks::Graphs::benchmarkProgressDir( parameters, "bfs" ),
         [ & ]()
         {
            ok = resolvePrecision( benchmark, parameters );
         } );
   }

   if( ! ok )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

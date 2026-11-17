// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <system_error>

#include <TNL/Config/ParameterContainer.h>
#include <TNL/MPI/Comm.h>
#include <TNL/MPI/Utils.h>
#include <TNL/String.h>

namespace TNL::Benchmarks::Graphs {

// Returns true when the process was started by an MPI launcher together with other processes
// (`mpirun -np N` of Open MPI, MPICH or Intel MPI, or `srun` with an MPI plugin, with N > 1).
// MPI is initialized only in that case: a single-process run does not depend on a working MPI
// installation (e.g. Open MPI refuses to initialize on a node with both CUDA and ROCm installed
// unless OMPI_MCA_accelerator is set), and it is not registered as an MPI task under `srun`,
// where a crash of an MPI task cancels the whole job step - including the shell script which
// resumes the benchmark after the crash.
[[nodiscard]] inline bool
startedAsSeveralMPIProcesses()
{
   const auto sizeFromEnv = []( const char* name )
   {
      const char* value = std::getenv( name );
      return value == nullptr ? 0 : std::atoi( value );
   };
   if( sizeFromEnv( "OMPI_COMM_WORLD_SIZE" ) > 1 || sizeFromEnv( "PMI_SIZE" ) > 1 )
      return true;
   // srun --mpi=pmix sets only the PMIx variables, the size of the job step is set by Slurm
   if( std::getenv( "PMIX_RANK" ) != nullptr && sizeFromEnv( "SLURM_STEP_NUM_TASKS" ) > 1 )
      return true;
   return false;
}

// MPI rank and size which work also when MPI was not initialized (see startedAsSeveralMPIProcesses()).
[[nodiscard]] inline int
mpiRank()
{
   return TNL::MPI::isInitialized() ? TNL::MPI::GetRank() : 0;
}

[[nodiscard]] inline int
mpiSize()
{
   return TNL::MPI::isInitialized() ? TNL::MPI::GetSize() : 1;
}

// Directory with the progress files of one benchmark (e.g. "bfs"). Each benchmark has its own
// subdirectory of --progress-dir, since the cells of different benchmarks may have equal keys
// (e.g. the TNL CSR kernel of SSSP and CC).
[[nodiscard]] inline std::string
benchmarkProgressDir( const TNL::Config::ParameterContainer& parameters, const std::string& benchmarkName )
{
   const std::string dir = parameters.getParameter< TNL::String >( "progress-dir" ).getString();
   return ( std::filesystem::path( dir ) / benchmarkName ).string();
}

// Graph name used for the progress file of the graph currently being benchmarked: --graph-name
// if set (always the case with --graph-list), the file name of --input-file otherwise. The file
// name is used instead of the whole path, because an absolute input-file path would make the
// progress file path discard the progress directory entirely (std::filesystem::path's operator/
// replaces the whole path when the right-hand side is absolute).
[[nodiscard]] inline std::string
progressGraphName( const TNL::Config::ParameterContainer& parameters )
{
   const TNL::String graphName = parameters.getParameter< TNL::String >( "graph-name" );
   if( ! graphName.empty() )
      return graphName.getString();
   return std::filesystem::path( parameters.getParameter< TNL::String >( "input-file" ).getString() ).filename().string();
}

// Tracks which (device, precision, orientation, solver, kernel, launch cfg, mode) sweep cells have
// already been timed and logged for the graph currently being benchmarked, so a resumed run can
// skip straight to the first not-yet-done cell instead of redoing the whole graph. The backing
// file is <dir>/<graphName>[.rankN].progress, one tab-separated key per completed cell. A cell is
// one entry of the sweep (e.g. one TNL segments type with one launch config and one mode, or one
// other solver like gapbs/Boost/Gunrock) - the GAP-prescribed trials within a single-source cell
// (e.g. BFS's 64 source vertices) are skipped/done together as one unit, not individually.
//
// When `worldSize > 1` (several MPI ranks, each bound to its own GPU, splitting one graph's GPU
// kernel sweep between them), each GPU-capable *kernel* (a TNL segments type name, or "Gunrock")
// is additionally owned by exactly one rank, chosen by hashing its name - isMine() reports
// ownership independently on every rank, with no coordination needed. CPU-only work (gapbs,
// Boost, and the TNL Sequential/Host device runs) does not go through isMine() at all: it is
// restricted to rank 0 by a plain rank-number check at its call sites instead.
class GraphBenchmarkProgress
{
public:
   GraphBenchmarkProgress( const std::string& dir, const std::string& graphName, int rank = 0, int worldSize = 1 )
   : filePath( filePathFor( dir, graphName, rank, worldSize ) ),
     rank( rank ),
     worldSize( worldSize )
   {
      std::ifstream file( filePath );
      for( std::string line; std::getline( file, line ); )
         if( ! line.empty() )
            done.insert( line );
   }

   // `device` is the device the cell runs on (e.g. TNL::getType< Device >()): the same kernel
   // with the same launch configuration is a different cell on each device.
   [[nodiscard]] bool
   isDone(
      const std::string& device,
      const std::string& precision,
      const std::string& orientation,
      const std::string& solver,
      const std::string& kernel,
      const std::string& launchConfig,
      const std::string& mode ) const
   {
      return done.count( key( device, precision, orientation, solver, kernel, launchConfig, mode ) ) > 0;
   }

   // Appends and flushes immediately: the benchmark process can be terminated at any time, so a
   // completed cell must be durable on disk before moving on to the next one, not just sitting in
   // a buffer.
   void
   markDone(
      const std::string& device,
      const std::string& precision,
      const std::string& orientation,
      const std::string& solver,
      const std::string& kernel,
      const std::string& launchConfig,
      const std::string& mode )
   {
      const std::string k = key( device, precision, orientation, solver, kernel, launchConfig, mode );
      done.insert( k );
      std::ofstream file( filePath, std::ios::app );
      file << k << '\n';
      file.flush();
   }

   // FNV-1a: a small, fully deterministic string hash. Used instead of std::hash<std::string>,
   // which the C++ standard does not guarantee to be stable across processes, builds or standard
   // library versions - every MPI rank must compute exactly the same hash for the same kernel
   // name to independently agree on who owns it, with no coordination at all.
   static std::uint64_t
   fnv1a( const std::string& s )
   {
      std::uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
      for( unsigned char c : s ) {
         h ^= c;
         h *= 0x100000001b3ULL;  // FNV prime
      }
      return h;
   }

   [[nodiscard]] bool
   isMine( const std::string& kernel ) const
   {
      if( worldSize <= 1 )
         return true;
      return static_cast< int >( fnv1a( kernel ) % static_cast< std::uint64_t >( worldSize ) ) == rank;
   }

private:
   static std::string
   key(
      const std::string& device,
      const std::string& precision,
      const std::string& orientation,
      const std::string& solver,
      const std::string& kernel,
      const std::string& launchConfig,
      const std::string& mode )
   {
      return device + '\t' + precision + '\t' + orientation + '\t' + solver + '\t' + kernel + '\t' + launchConfig + '\t' + mode;
   }

   static std::filesystem::path
   filePathFor( const std::string& dir, const std::string& graphName, int rank, int worldSize )
   {
      const std::string extension = worldSize > 1 ? "rank" + std::to_string( rank ) + ".progress" : std::string( "progress" );
      const std::filesystem::path path = std::filesystem::path( dir ) / ( graphName + "." + extension );
      std::filesystem::create_directories( path.parent_path() );
      return path;
   }

   std::filesystem::path filePath;
   std::set< std::string > done;
   int rank;
   int worldSize;
};

// Deletes a graph's progress files (every rank's when `worldSize > 1`), if present. Called once a
// graph has been fully processed - every precision, every rank - right before it is removed from
// the graph list: its progress files are no longer needed at that point.
inline void
removeGraphProgress( const std::string& dir, const std::string& graphName, int worldSize = 1 )
{
   std::error_code ec;
   if( worldSize <= 1 ) {
      std::filesystem::remove( std::filesystem::path( dir ) / ( graphName + ".progress" ), ec );
      return;
   }
   for( int rank = 0; rank < worldSize; rank++ )
      std::filesystem::remove(
         std::filesystem::path( dir ) / ( graphName + ".rank" + std::to_string( rank ) + ".progress" ), ec );
}

}  // namespace TNL::Benchmarks::Graphs

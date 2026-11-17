// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <TNL/Config/ParameterContainer.h>
#include <TNL/MPI/Comm.h>
#include <TNL/String.h>

#include "GraphBenchmarkProgress.h"

namespace TNL::Benchmarks::Graphs {

// Derives a graph name from its path relative to `basePath`, used both for the "graph name" log
// column and for naming the graph's progress file.
inline std::string
graphNameFromPath( const std::string& path, const std::string& basePath )
{
   std::string name = path;

   std::string base = basePath;
   while( ! base.empty() && base.back() == '/' )
      base.pop_back();
   const std::string prefix = base + "/";
   if( name.rfind( prefix, 0 ) == 0 )
      name = name.substr( prefix.size() );
   if( ! name.empty() && name.front() == '/' )
      name = name.substr( 1 );

   return name;
}

// Manages a flat, newline-separated list of remaining graph file paths. The list is read fully
// into memory once; removeFirst() rewrites the file with the first entry gone.
//
// A graph must be removed only once it is FULLY processed, never optimistically at the start:
// the benchmark process can be killed at any time (segfault, SIGKILL, wall-clock limit, ...), and
// as long as a graph stays at the head of the list, re-running with the same list file (and the
// same --progress-dir, see GraphBenchmarkProgress.h) resumes it instead of silently skipping it.
class GraphList
{
public:
   explicit GraphList( std::string path )
   : path( std::move( path ) )
   {
      std::ifstream file( this->path );
      for( std::string line; std::getline( file, line ); )
         if( ! line.empty() )
            entries.push_back( line );
   }

   [[nodiscard]] bool
   empty() const
   {
      return entries.empty();
   }

   [[nodiscard]] const std::string&
   front() const
   {
      return entries.front();
   }

   [[nodiscard]] std::size_t
   size() const
   {
      return entries.size();
   }

   // Removes the first entry and durably rewrites the list file: written to a temporary file,
   // flushed and closed, then renamed over the original - the rename is atomic, so a crash right
   // after this call never leaves the list file truncated or half-written.
   void
   removeFirst()
   {
      entries.erase( entries.begin() );

      const std::string tmpPath = path + ".tmp";
      {
         std::ofstream file( tmpPath, std::ios::trunc );
         for( const std::string& entry : entries )
            file << entry << '\n';
         file.flush();
      }
      std::filesystem::rename( tmpPath, path );
   }

private:
   std::string path;
   std::vector< std::string > entries;
};

// Loads `listFile` and calls `processGraph()` once for every remaining graph, in order,
// replacing a former per-graph shell loop. Before each call, "input-file"/"graph-name" are set on
// `parameters` (graph name derived from the path via graphNameFromPath()). A graph is removed
// from the list - and its progress file(s) deleted, when `progressDir` is not empty - only after
// `processGraph()` returns, whether it threw a (caught) exception or not: a crash that takes the
// whole process down leaves the graph at the head of the list, so re-running with the same
// listFile/progressDir resumes it instead of silently skipping it.
//
// When running as more than one MPI rank (several ranks splitting one graph's GPU kernel sweep
// between them, each bound to its own GPU), every rank must agree on which graph is "current" and
// a graph may only be removed once *every* rank has finished its own share of it - so this takes
// a different loop shape in that case: the list is re-read from disk on every iteration (instead
// of kept purely in memory) so all ranks observe the same state, a barrier after processGraph()
// confirms every rank is done with the current graph, and only rank 0 then mutates the list file
// and deletes the (every-rank) progress files for it, followed by a second barrier so nobody
// re-reads the list before rank 0's write has landed. With a single rank (the default - no
// `mpirun`), none of this applies and the loop is exactly as before.
template< typename ProcessGraph >
void
runOverGraphList(
   TNL::Config::ParameterContainer& parameters,
   const std::string& listFile,
   const std::string& basePath,
   const std::string& progressDir,
   ProcessGraph processGraph )
{
   const int rank = mpiRank();
   const int worldSize = mpiSize();

   if( worldSize <= 1 ) {
      GraphList graphList( listFile );
      const std::size_t total = graphList.size();

      while( ! graphList.empty() ) {
         const std::string path = graphList.front();
         const std::string name = graphNameFromPath( path, basePath );
         std::cout << "\n==> " << ( total - graphList.size() + 1 ) << " / " << total << " : graph " << name << " (" << path
                   << ")" << std::endl;

         parameters.setParameter( "input-file", TNL::String( path ) );
         parameters.setParameter( "graph-name", TNL::String( name ) );
         try {
            processGraph();
         }
         catch( const std::exception& e ) {
            std::cerr << "Unable to benchmark graph " << name << ": " << e.what() << std::endl;
         }

         if( ! progressDir.empty() )
            removeGraphProgress( progressDir, name );
         graphList.removeFirst();
      }
      return;
   }

   while( true ) {
      GraphList graphList( listFile );
      if( graphList.empty() )
         break;
      const std::string path = graphList.front();
      const std::string name = graphNameFromPath( path, basePath );
      if( rank == 0 )
         std::cout << "\n==> " << graphList.size() << " graphs left : graph " << name << " (" << path << ")" << std::endl;

      parameters.setParameter( "input-file", TNL::String( path ) );
      parameters.setParameter( "graph-name", TNL::String( name ) );
      try {
         processGraph();
      }
      catch( const std::exception& e ) {
         std::cerr << "Rank " << rank << ": unable to benchmark graph " << name << ": " << e.what() << std::endl;
      }

      // every rank must finish its own share of this graph before anyone moves on
      TNL::MPI::Barrier();
      if( rank == 0 ) {
         if( ! progressDir.empty() )
            removeGraphProgress( progressDir, name, worldSize );
         graphList.removeFirst();
      }
      // make sure rank 0's write above has landed before any rank re-reads the list
      TNL::MPI::Barrier();
   }
}

// Benchmarks the single graph given by --input-file (without --graph-list) and deletes its
// progress file(s) afterwards, as runOverGraphList() does for each graph of the list: otherwise a
// later run on the same graph would find every cell done and silently measure nothing. A crash
// that takes the whole process down leaves the progress file(s) in place, so re-running the same
// command resumes the graph.
template< typename ProcessGraph >
void
runSingleGraph( const TNL::Config::ParameterContainer& parameters, const std::string& progressDir, ProcessGraph processGraph )
{
   const int worldSize = mpiSize();
   const std::string name = progressGraphName( parameters );
   try {
      processGraph();
   }
   catch( const std::exception& e ) {
      std::cerr << "Rank " << mpiRank() << ": unable to benchmark graph " << name << ": " << e.what() << std::endl;
   }

   // every rank must finish its own share of the graph before its progress files are deleted
   if( worldSize > 1 )
      TNL::MPI::Barrier();
   if( mpiRank() == 0 && ! progressDir.empty() )
      removeGraphProgress( progressDir, name, worldSize );
}

}  // namespace TNL::Benchmarks::Graphs

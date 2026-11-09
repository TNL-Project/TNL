// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

// The kernels of the GAP reference code are defined in the sources of its benchmark drivers together with their
// main functions, which are renamed here. The include directory of gapbs (its src directory) is set by CMake.

#include "GapbsKernels.h"

#include <fstream>
#include <stdexcept>

#define main gapbs_bfs_main
#include <bfs.cc>
#undef main
#define main gapbs_sssp_main
#include <sssp.cc>
#undef main
#define main gapbs_cc_main
#include <cc.cc>
#undef main

namespace TNL::Benchmarks::Graphs::Gapbs {

namespace {

// The gapbs reader terminates the program on errors, so the file is checked first.
void
checkFile( const std::string& fileName, const std::string& suffix )
{
   if( fileName.size() < suffix.size() || fileName.compare( fileName.size() - suffix.size(), suffix.size(), suffix ) != 0 )
      throw std::invalid_argument( "gapbs: the file " + fileName + " must have the suffix " + suffix );
   std::ifstream file( fileName, std::ios::binary );
   if( ! file )
      throw std::runtime_error( "gapbs: cannot open the file " + fileName );
}

}  // namespace

struct BFS::Impl
{
   Graph graph;
   pvector< NodeID > parents;
};

BFS::BFS( const std::string& sgFile )
: impl( std::make_unique< Impl >() )
{
   checkFile( sgFile, ".sg" );
   Reader< NodeID, NodeID, WeightT, true > reader( sgFile );
   impl->graph = reader.ReadSerializedGraph();
}

BFS::~BFS() = default;

std::int64_t
BFS::getVertexCount() const
{
   return impl->graph.num_nodes();
}

void
BFS::release()
{
   impl->parents = pvector< NodeID >();
}

void
BFS::run( std::int32_t source )
{
   impl->parents = DOBFS( impl->graph, source );
}

void
BFS::getParents( std::vector< std::int32_t >& parents ) const
{
   parents.assign( impl->parents.begin(), impl->parents.end() );
}

struct SSSP::Impl
{
   WGraph graph;
   pvector< WeightT > distances;
};

SSSP::SSSP( const std::string& wsgFile )
: impl( std::make_unique< Impl >() )
{
   checkFile( wsgFile, ".wsg" );
   Reader< NodeID, WNode, WeightT, true > reader( wsgFile );
   impl->graph = reader.ReadSerializedGraph();
}

SSSP::~SSSP() = default;

std::int64_t
SSSP::getVertexCount() const
{
   return impl->graph.num_nodes();
}

void
SSSP::release()
{
   impl->distances = pvector< WeightT >();
}

void
SSSP::run( std::int32_t source, std::int32_t delta )
{
   impl->distances = DeltaStep( impl->graph, source, delta );
}

void
SSSP::getDistances( std::vector< std::int64_t >& distances ) const
{
   distances.resize( impl->distances.size() );
   for( std::size_t i = 0; i < distances.size(); i++ )
      distances[ i ] = impl->distances[ i ] == kDistInf ? -1 : impl->distances[ i ];
}

struct CC::Impl
{
   Graph graph;
   pvector< NodeID > components;
};

CC::CC( const std::string& sgFile )
: impl( std::make_unique< Impl >() )
{
   checkFile( sgFile, ".sg" );
   Reader< NodeID, NodeID, WeightT, true > reader( sgFile );
   impl->graph = reader.ReadSerializedGraph();
}

CC::~CC() = default;

std::int64_t
CC::getVertexCount() const
{
   return impl->graph.num_nodes();
}

void
CC::release()
{
   impl->components = pvector< NodeID >();
}

void
CC::run()
{
   impl->components = Afforest( impl->graph );
}

void
CC::getComponents( std::vector< std::int32_t >& components ) const
{
   components.assign( impl->components.begin(), impl->components.end() );
}

}  // namespace TNL::Benchmarks::Graphs::Gapbs

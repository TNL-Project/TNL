// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

// The kernels of the GAP reference code are defined in the sources of its benchmark drivers together with their
// main functions, which are renamed here. The include directory of gapbs (its src directory) is set by CMake.

#include "GapbsKernels.h"

#include <algorithm>
#include <fstream>
#include <limits>
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

// Builds the graph as the gapbs builder (see BuilderBase::SquishCSR): the neighbors are sorted, the duplicate edges
// and self-loops are removed. The incoming edges of a directed graph are sorted by the order of the filling.
Graph
makeGraph(
   std::int64_t vertexCount,
   const std::vector< std::int64_t >& offsets,
   const std::vector< std::int32_t >& columns,
   bool directed )
{
   if( vertexCount > std::numeric_limits< NodeID >::max() )
      throw std::invalid_argument( "gapbs: the graph has more vertices than its 32-bit vertex IDs can address" );
   if( offsets.size() != static_cast< std::size_t >( vertexCount ) + 1 )
      throw std::invalid_argument( "gapbs: the number of the CSR offsets does not match the number of vertices" );
   const NodeID n = vertexCount;

   // squish the neighbors of every vertex in a copy of the column indexes
   std::vector< NodeID > squished( columns.begin(), columns.end() );
   pvector< SGOffset > degrees( n );
#pragma omp parallel for schedule( dynamic, 1024 )
   for( NodeID v = 0; v < n; v++ ) {
      NodeID* begin = squished.data() + offsets[ v ];
      NodeID* end = squished.data() + offsets[ v + 1 ];
      end = std::remove_if(
         begin,
         end,
         [ & ]( NodeID w )
         {
            return w < 0 || w >= n || w == v;
         } );
      std::sort( begin, end );
      end = std::unique( begin, end );
      degrees[ v ] = end - begin;
   }

   pvector< SGOffset > outOffsets( n + 1 );
   outOffsets[ 0 ] = 0;
   for( NodeID v = 0; v < n; v++ )
      outOffsets[ v + 1 ] = outOffsets[ v ] + degrees[ v ];
   auto* outNeighbors = new NodeID[ outOffsets[ n ] ];
#pragma omp parallel for schedule( dynamic, 1024 )
   for( NodeID v = 0; v < n; v++ )
      std::copy_n( squished.data() + offsets[ v ], degrees[ v ], outNeighbors + outOffsets[ v ] );
   NodeID** outIndex = Graph::GenIndex( outOffsets, outNeighbors );
   if( ! directed )
      return { n, outIndex, outNeighbors };

   // the incoming edges, filled in the order of the sources, so they are sorted
   pvector< SGOffset > inOffsets( n + 1 );
   std::fill( inOffsets.begin(), inOffsets.end(), 0 );
   for( SGOffset k = 0; k < outOffsets[ n ]; k++ )
      inOffsets[ outNeighbors[ k ] + 1 ]++;
   for( NodeID v = 0; v < n; v++ )
      inOffsets[ v + 1 ] += inOffsets[ v ];
   auto* inNeighbors = new NodeID[ inOffsets[ n ] ];
   std::vector< SGOffset > position( inOffsets.begin(), inOffsets.end() - 1 );
   for( NodeID v = 0; v < n; v++ )
      for( SGOffset k = outOffsets[ v ]; k < outOffsets[ v + 1 ]; k++ )
         inNeighbors[ position[ outNeighbors[ k ] ]++ ] = v;
   NodeID** inIndex = Graph::GenIndex( inOffsets, inNeighbors );
   return { n, outIndex, outNeighbors, inIndex, inNeighbors };
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

BFS::BFS(
   std::int64_t vertexCount,
   const std::vector< std::int64_t >& offsets,
   const std::vector< std::int32_t >& columns,
   bool directed )
: impl( std::make_unique< Impl >() )
{
   impl->graph = makeGraph( vertexCount, offsets, columns, directed );
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

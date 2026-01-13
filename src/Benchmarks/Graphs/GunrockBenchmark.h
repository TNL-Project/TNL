// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#ifdef HAVE_GUNROCK
   #include <thrust/device_vector.h>
   #include <gunrock/graph/graph.hxx>
   #include <gunrock/algorithms/bfs.hxx>
   #include <gunrock/algorithms/sssp.hxx>
#endif

#include <TNL/Algorithms/copy.h>

template< typename Value = double, typename Index = int >
struct GunrockBenchmark
{
   using IndexType = Index;
   using ValueType = Value;

#ifdef HAVE_GUNROCK
   template< typename HostGraphType >
   static auto
   convertToGunrockGraph( const HostGraphType& hostGraph )
   {
      using GraphType = TNL::Graphs::Graph< ValueType, TNL::Devices::Cuda, IndexType >;
      GraphType graph;
      graph = hostGraph;
      auto adjacencyMatrix = graph.getAdjacencyMatrix();

      const auto& adjacencyMatrix = hostGraph.getAdjacencyMatrix();

      TNL::Containers::Vector< IndexType > rowIndices( adjacencyMatrix.getValues().getSize() );
      TNL::Containers::Vector< IndexType > columnOffsets( adjacencyMatrix.getColumns() + 1 );

      auto graph = gunrock::graph::build::from_csr< gunrock::memory_space_t::device, gunrock::graph::view_t::csr >(
         adjacencyMatrix.getRows(),                             // rows
         adjacencyMatrix.getColumns(),                          // columns
         adjacencyMatrix.getValues().getSize(),                 // nonzeros
         adjacencyMatrix.getSegments().getOffsets().getData(),  // row_offsets
         adjacencyMatrix.getColumnIndexes().getData(),          // column_indices
         adjacencyMatrix.getValues().getData(),                 // values
         rowIndices.getData(),                                  // row_indices
         columnOffsets.getData()                                // column_offsets
      );
      return graph;
   }
#endif

   template< typename Graph >
   void
   breadthFirstSearch( TNL::Benchmarks::Benchmark<>& benchmark,
                       Graph& graph,
                       Index start,
                       Index size,
                       std::vector< Index >& distances )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< typename Graph::vertex_type > d_distances( size );
      thrust::device_vector< typename Graph::vertex_type > d_predecessors( size );

      auto bfs_gunrock = [ & ]() mutable
      {
         gunrock::bfs::run( graph, start, d_distances.data().get(), d_predecessors.data().get() );
      };
      benchmark.time< TNL::Devices::Cuda >( "cuda", bfs_gunrock );
      TNL_ASSERT_EQ( d_distances.size(), distances.size(), "Size mismatch in Gunrock BFS distances." );
      thrust::copy( d_distances.begin(), d_distances.end(), distances.begin() );
#endif
   }

   template< typename Graph >
   void
   singleSourceShortestPath( TNL::Benchmarks::Benchmark<>& benchmark,
                             Graph& graph,
                             Index start,
                             Index size,
                             std::vector< Value >& distances )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< ValueType > d_distances( size );
      thrust::device_vector< IndexType > d_predecessors( size );

      auto bfs_gunrock = [ & ]() mutable
      {
         gunrock::sssp::run( graph, start, d_distances.data().get(), d_predecessors.data().get() );
      };
      benchmark.time< TNL::Devices::Cuda >( "cuda", bfs_gunrock );
      TNL_ASSERT_EQ( d_distances.size(), distances.size(), "Size mismatch in Gunrock BFS distances." );
      thrust::copy( d_distances.begin(), d_distances.end(), distances.begin() );
#endif
   }
};

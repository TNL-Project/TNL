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

template< typename Real = double, typename Index = int >
struct GunrockBenchmark
{
   using IndexType = Index;
   using RealType = Real;

#ifdef HAVE_GUNROCK
   template< typename HostGraphType >
   static auto
   convertToGunrockGraph( const HostGraphType& hostGraph )
   {
      std::vector< IndexType > rowOffsets;
      std::vector< IndexType > columnIndices;
      std::vector< RealType > values;

      const auto& adjacencyMatrix = hostGraph.getAdjacencyMatrix();
      TNL::Algorithms::copy( rowOffsets, adjacencyMatrix.getSegments().getOffsets() );
      TNL::Algorithms::copy( columnIndices, adjacencyMatrix.getColumnIndexes() );
      TNL::Algorithms::copy( values, adjacencyMatrix.getValues() );

      thrust::device_vector< IndexType > d_rowOffsets( adjacencyMatrix.getRows() + 1 );
      thrust::device_vector< IndexType > d_columnIndices( adjacencyMatrix.getColumnIndexes().getSize() );
      thrust::device_vector< RealType > d_values( adjacencyMatrix.getValues().getSize() );
      thrust::device_vector< IndexType > d_rowIndices( adjacencyMatrix.getValues().getSize() );
      thrust::device_vector< IndexType > d_columnOffsets( adjacencyMatrix.getColumns() + 1 );

      thrust::copy( rowOffsets.begin(), rowOffsets.end(), d_rowOffsets.begin() );
      thrust::copy( columnIndices.begin(), columnIndices.end(), d_columnIndices.begin() );
      thrust::copy( values.begin(), values.end(), d_values.begin() );

      auto graph = gunrock::graph::build::from_csr< gunrock::memory_space_t::device, gunrock::graph::view_t::csr >(
         adjacencyMatrix.getRows(),              // rows
         adjacencyMatrix.getColumns(),           // columns
         adjacencyMatrix.getValues().getSize(),  // nonzeros
         d_rowOffsets.data().get(),              // row_offsets
         d_columnIndices.data().get(),           // column_indices
         d_values.data().get(),                  // values
         d_rowIndices.data().get(),              // row_indices
         d_columnOffsets.data().get()            // column_offsets
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
                             std::vector< Real >& distances )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< Real > d_distances( size );
      thrust::device_vector< Index > d_predecessors( size );

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

// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Benchmarks/Benchmark.h>
#include <vector>
#ifdef HAVE_GUNROCK
   #include <thrust/device_vector.h>
   #include <gunrock/algorithms/bfs.hxx>
   #include <gunrock/algorithms/sssp.hxx>
   #include <gunrock/graph/build.hxx>
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
      const auto& adjacencyMatrix = hostGraph.getAdjacencyMatrix();
      const auto& segments = adjacencyMatrix.getSegments();
      const auto& offsets = segments.getOffsets();
      const auto& columnIndexes = adjacencyMatrix.getColumnIndexes();
      const auto& values = adjacencyMatrix.getValues();

      const IndexType numRows = adjacencyMatrix.getRows();
      const IndexType numCols = adjacencyMatrix.getColumns();
      const IndexType numNonzeros = values.getSize();

      using csr_t = gunrock::format::csr_t< gunrock::memory_space_t::device, IndexType, IndexType, ValueType >;
      using graph_type = decltype( gunrock::graph::build< gunrock::memory_space_t::device, IndexType, IndexType, ValueType >(
         std::declval< gunrock::graph::graph_properties_t >(), std::declval< csr_t& >() ) );

      struct GunrockGraphHolder
      {
         csr_t csr;
         graph_type graph;
      };

      GunrockGraphHolder holder;
      holder.csr = csr_t( numRows, numCols, numNonzeros );

      thrust::host_vector< IndexType > h_offsets( numRows + 1 );
      TNL::Algorithms::copy< TNL::Devices::Host, TNL::Devices::Host >( h_offsets.data(), offsets.getData(), numRows + 1 );
      holder.csr.row_offsets = h_offsets;

      thrust::host_vector< IndexType > h_columns( numNonzeros );
      TNL::Algorithms::copy< TNL::Devices::Host, TNL::Devices::Host >( h_columns.data(), columnIndexes.getData(), numNonzeros );
      holder.csr.column_indices = h_columns;

      thrust::host_vector< ValueType > h_values( numNonzeros );
      TNL::Algorithms::copy< TNL::Devices::Host, TNL::Devices::Host >( h_values.data(), values.getData(), numNonzeros );
      holder.csr.nonzero_values = h_values;

      auto properties = gunrock::graph::graph_properties_t{};
      properties.directed = true;

      holder.graph =
         gunrock::graph::build< gunrock::memory_space_t::device, IndexType, IndexType, ValueType >( properties, holder.csr );

      return holder;
   }
#endif

   template< typename Graph >
   void
   breadthFirstSearch(
      TNL::Benchmarks::Benchmark& benchmark,
      Graph& graph,
      Index start,
      Index size,
      std::vector< Index >& distances )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< typename Graph::vertex_type > d_distances( size );
      thrust::device_vector< typename Graph::vertex_type > d_predecessors( size );

      typename Graph::vertex_type source = start;
      auto bfs_gunrock = [ & ]() mutable
      {
         gunrock::bfs::run( graph, source, d_distances.data().get(), d_predecessors.data().get() );
      };
      benchmark.time< TNL::Devices::Cuda >( "cuda", bfs_gunrock );
      TNL_ASSERT_EQ( d_distances.size(), distances.size(), "Size mismatch in Gunrock BFS distances." );
      thrust::copy( d_distances.begin(), d_distances.end(), distances.begin() );
#endif
   }

   template< typename Graph >
   void
   singleSourceShortestPath(
      TNL::Benchmarks::Benchmark& benchmark,
      Graph& graph,
      Index start,
      Index size,
      std::vector< Value >& distances )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< ValueType > d_distances( size );
      thrust::device_vector< IndexType > d_predecessors( size );

      typename Graph::vertex_type source = start;
      auto sssp_gunrock = [ & ]() mutable
      {
         gunrock::sssp::run( graph, source, d_distances.data().get(), d_predecessors.data().get() );
      };
      benchmark.time< TNL::Devices::Cuda >( "cuda", sssp_gunrock );
      TNL_ASSERT_EQ( d_distances.size(), distances.size(), "Size mismatch in Gunrock SSSP distances." );
      thrust::copy( d_distances.begin(), d_distances.end(), distances.begin() );
#endif
   }
};

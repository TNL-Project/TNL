// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Benchmarks/Benchmark.h>
#include <algorithm>
#include <vector>
#ifdef HAVE_GUNROCK
   #include <thrust/device_vector.h>
   #include <gunrock/algorithms/bfs.hxx>
   #include <gunrock/algorithms/sssp.hxx>
   #include <gunrock/graph/build.hxx>
#endif

#include <TNL/Algorithms/copy.h>
#include <list>
#include <string>
#include <utility>

template< typename Value = double, typename Index = int >
struct GunrockBenchmark
{
   using IndexType = Index;
   using ValueType = Value;

   /**
    * Gunrock's advance load-balancing strategies that are actually runtime
    * dispatchable (see gunrock::operators::load_balance_t and
    * operators::advance::execute_runtime).  warp_mapped, bucketing and
    * work_stealing are declared in Gunrock but are unimplemented (WIP), so
    * they are intentionally left out here.  merge_path_v2 is also excluded:
    * its advance kernel (merge_path_v2.hxx) unconditionally dumps the input
    * and output frontiers to stdout with no way to silence it, and it was
    * observed to produce incorrect BFS/SSSP distances on undirected graphs.
    */
   enum class LoadBalance : std::uint8_t
   {
      ThreadMapped,
      BlockMapped,
      MergePath
   };

   static std::list< std::pair< LoadBalance, std::string > >
   loadBalanceConfigurations()
   {
      return {
         { LoadBalance::ThreadMapped, "thread_mapped" },
         { LoadBalance::BlockMapped, "block_mapped" },
         { LoadBalance::MergePath, "merge_path" },
      };
   }

#ifdef HAVE_GUNROCK
   static gunrock::operators::load_balance_t
   toGunrockLoadBalance( LoadBalance loadBalance )
   {
      switch( loadBalance ) {
         case LoadBalance::ThreadMapped:
            return gunrock::operators::load_balance_t::thread_mapped;
         case LoadBalance::MergePath:
            return gunrock::operators::load_balance_t::merge_path;
         case LoadBalance::BlockMapped:
         default:
            return gunrock::operators::load_balance_t::block_mapped;
      }
   }
#endif

#ifdef HAVE_GUNROCK
   template< typename HostGraphType >
   static auto
   convertToGunrockGraph( const HostGraphType& hostGraph )
   {
      const auto& adjacencyMatrix = hostGraph.getAdjacencyMatrix();
      const auto& segments = adjacencyMatrix.getSegments();
      const auto& offsets = segments.getOffsets();
      const auto& columnIndexes = adjacencyMatrix.getColumnIndexes();

      const IndexType numRows = adjacencyMatrix.getRows();
      const IndexType numCols = adjacencyMatrix.getColumns();
      const IndexType numNonzeros = adjacencyMatrix.getNonzeroElementsCount();

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
      if constexpr( HostGraphType::AdjacencyMatrixType::isBinary() ) {
         // Binary matrices don't store any values -- Gunrock still expects a
         // nonzero_values array, so fill it with a constant "edge present" value.
         std::fill( h_values.begin(), h_values.end(), ValueType{ 1 } );
      }
      else {
         const auto& values = adjacencyMatrix.getValues();
         TNL::Algorithms::copy< TNL::Devices::Host, TNL::Devices::Host >( h_values.data(), values.getData(), numNonzeros );
      }
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
      std::vector< Index >& distances,
      LoadBalance loadBalance = LoadBalance::BlockMapped )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< typename Graph::vertex_type > d_distances( size );
      thrust::device_vector< typename Graph::vertex_type > d_predecessors( size );

      typename Graph::vertex_type source = start;
      auto gunrockLoadBalance = toGunrockLoadBalance( loadBalance );
      auto bfs_gunrock = [ & ]() mutable
      {
         // A fresh context must be constructed on every call: benchmark.time()
         // invokes this lambda repeatedly (warmup + measured loops), and
         // Gunrock's enactor is not safe to reuse the same multi_context_t
         // across independent runs (matches gunrock::bfs::run's own default
         // argument, which likewise constructs a new context per call).
         auto context = std::shared_ptr< gunrock::gcuda::multi_context_t >( new gunrock::gcuda::multi_context_t( 0 ) );
         gunrock::bfs::run( graph, source, d_distances.data().get(), d_predecessors.data().get(), context, gunrockLoadBalance );
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
      std::vector< Value >& distances,
      LoadBalance loadBalance = LoadBalance::BlockMapped )
   {
#ifdef HAVE_GUNROCK
      thrust::device_vector< ValueType > d_distances( size );
      thrust::device_vector< IndexType > d_predecessors( size );

      typename Graph::vertex_type source = start;
      auto gunrockLoadBalance = toGunrockLoadBalance( loadBalance );
      auto sssp_gunrock = [ & ]() mutable
      {
         // See breadthFirstSearch above: a fresh context per call is required.
         auto context = std::shared_ptr< gunrock::gcuda::multi_context_t >( new gunrock::gcuda::multi_context_t( 0 ) );
         gunrock::sssp::run(
            graph, source, d_distances.data().get(), d_predecessors.data().get(), context, gunrockLoadBalance );
      };
      benchmark.time< TNL::Devices::Cuda >( "cuda", sssp_gunrock );
      TNL_ASSERT_EQ( d_distances.size(), distances.size(), "Size mismatch in Gunrock SSSP distances." );
      thrust::copy( d_distances.begin(), d_distances.end(), distances.begin() );
#endif
   }
};

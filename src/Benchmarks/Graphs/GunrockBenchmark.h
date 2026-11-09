// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Benchmarks/Benchmark.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#ifdef HAVE_GUNROCK
   #include <thrust/device_vector.h>
   #include <gunrock/algorithms/algorithms.hxx>
   #include <gunrock/algorithms/sssp.hxx>
   #include <gunrock/graph/build.hxx>
#endif

#include <TNL/Algorithms/copy.h>
#include <list>
#include <string>
#include <utility>

#ifdef HAVE_GUNROCK
/**
 * BFS computing the parents (predecessors) of the vertices, implemented with the operators of Gunrock.
 *
 * The BFS of Gunrock (gunrock/algorithms/bfs.hxx) computes only the distances; its result has a predecessors
 * array, but it is not filled ("@todo: implement this"). The GAP benchmark prescribes BFS computing the parents,
 * so this is the BFS of Gunrock extended by one store: the thread whose atomic minimum discovers a vertex records
 * the source of the edge as its parent. Exactly one thread discovers each vertex, since the other threads of the
 * same iteration see the new distance, so the parent is a vertex of the previous level and the result is a valid
 * BFS tree. Everything else (frontiers, advance and filter operators, load balancing) is identical to Gunrock.
 */
namespace gunrock_bfs_with_predecessors {

template< typename vertex_t >
struct param_t
{
   vertex_t single_source;
   gunrock::options_t options;
};

template< typename vertex_t >
struct result_t
{
   vertex_t* distances;
   vertex_t* predecessors;
};

template< typename graph_t, typename param_type, typename result_type >
struct problem_t : gunrock::problem_t< graph_t >
{
   param_type param;
   result_type result;

   problem_t(
      graph_t& G,
      param_type& _param,
      result_type& _result,
      std::shared_ptr< gunrock::gcuda::multi_context_t > _context )
   : gunrock::problem_t< graph_t >( G, _context ),
     param( _param ),
     result( _result )
   {}

   using vertex_t = typename graph_t::vertex_type;
   using edge_t = typename graph_t::edge_type;
   using weight_t = typename graph_t::weight_type;

   void
   init() override
   {}

   void
   reset() override
   {
      auto policy = this->context->get_context( 0 )->execution_policy();
      auto n_vertices = this->get_graph().get_number_of_vertices();
      const vertex_t source = this->param.single_source;
      auto d_distances = thrust::device_pointer_cast( this->result.distances );
      thrust::fill( policy, d_distances, d_distances + n_vertices, std::numeric_limits< vertex_t >::max() );
      thrust::fill( policy, d_distances + source, d_distances + source + 1, 0 );
      auto d_predecessors = thrust::device_pointer_cast( this->result.predecessors );
      thrust::fill( policy, d_predecessors, d_predecessors + n_vertices, -1 );
      thrust::fill( policy, d_predecessors + source, d_predecessors + source + 1, source );
   }
};

template< typename problem_t >
struct enactor_t : gunrock::enactor_t< problem_t >
{
   enactor_t( problem_t* _problem, std::shared_ptr< gunrock::gcuda::multi_context_t > _context )
   : gunrock::enactor_t< problem_t >( _problem, _context )
   {}

   using vertex_t = typename problem_t::vertex_t;
   using edge_t = typename problem_t::edge_t;
   using weight_t = typename problem_t::weight_t;
   using frontier_t = typename gunrock::enactor_t< problem_t >::frontier_t;

   void
   prepare_frontier( frontier_t* f, gunrock::gcuda::multi_context_t& context ) override
   {
      f->push_back( this->get_problem()->param.single_source );
   }

   void
   loop( gunrock::gcuda::multi_context_t& context ) override
   {
      auto E = this->get_enactor();
      auto P = this->get_problem();
      auto G = P->get_graph();

      auto distances = P->result.distances;
      auto predecessors = P->result.predecessors;
      auto iteration = this->iteration;

      auto search = [ distances, predecessors, iteration ] __host__ __device__(
                       vertex_t const& source, vertex_t const& neighbor, edge_t const& edge, weight_t const& weight ) -> bool
      {
         auto old_distance = gunrock::math::atomic::min( &distances[ neighbor ], iteration + 1 );
         if( iteration + 1 < old_distance ) {
            predecessors[ neighbor ] = source;
            return true;
         }
         return false;
      };

      auto remove_invalids = [] __host__ __device__( vertex_t const& vertex ) -> bool
      {
         return true;
      };

      gunrock::operators::advance::execute_runtime( G, E, search, P->param.options.advance_load_balance, context );
      if( P->param.options.enable_filter )
         gunrock::operators::filter::execute_runtime( G, E, remove_invalids, P->param.options.filter_algorithm, context );
   }
};

template< typename graph_t >
float
run(
   graph_t& G,
   param_t< typename graph_t::vertex_type >& param,
   result_t< typename graph_t::vertex_type >& result,
   std::shared_ptr< gunrock::gcuda::multi_context_t > context )
{
   using vertex_t = typename graph_t::vertex_type;
   using problem_type = problem_t< graph_t, param_t< vertex_t >, result_t< vertex_t > >;
   using enactor_type = enactor_t< problem_type >;

   problem_type problem( G, param, result, context );
   problem.init();
   problem.reset();

   enactor_type enactor( &problem, context );
   return enactor.enact();
}

}  // namespace gunrock_bfs_with_predecessors
#endif

/**
 * Wrappers of the Gunrock BFS and SSSP.
 *
 * Gunrock uses 32-bit vertex indexes; the edge offsets have the TNL index type, so that graphs with more than
 * 2^31 edges can be benchmarked with a 64-bit TNL index.
 */
template< typename Value = double, typename Index = int >
struct GunrockBenchmark
{
   using IndexType = Index;
   using ValueType = Value;
   using VertexType = int;
   using EdgeType = Index;

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

      using csr_t = gunrock::format::csr_t< gunrock::memory_space_t::device, VertexType, EdgeType, ValueType >;
      // note that gunrock::graph::build takes the edge type before the vertex type, unlike csr_t
      using graph_type = decltype( gunrock::graph::build< gunrock::memory_space_t::device, EdgeType, VertexType, ValueType >(
         std::declval< gunrock::graph::graph_properties_t >(), std::declval< csr_t& >() ) );

      struct GunrockGraphHolder
      {
         csr_t csr;
         graph_type graph;
      };

      GunrockGraphHolder holder;
      holder.csr = csr_t( numRows, numCols, numNonzeros );

      thrust::host_vector< EdgeType > h_offsets( numRows + 1 );
      TNL::Algorithms::copy< TNL::Devices::Host, TNL::Devices::Host >( h_offsets.data(), offsets.getData(), numRows + 1 );
      holder.csr.row_offsets = h_offsets;

      thrust::host_vector< VertexType > h_columns( numNonzeros );
      std::copy( columnIndexes.getData(), columnIndexes.getData() + numNonzeros, h_columns.begin() );
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
         gunrock::graph::build< gunrock::memory_space_t::device, EdgeType, VertexType, ValueType >( properties, holder.csr );

      return holder;
   }

   /**
    * \brief BFS with predecessors from one source, see \ref gunrock_bfs_with_predecessors.
    *
    * The output vectors are resized here, so their allocation is a part of the measured time.
    */
   template< typename Graph >
   static void
   breadthFirstSearch(
      Graph& graph,
      VertexType source,
      VertexType vertexCount,
      thrust::device_vector< VertexType >& distances,
      thrust::device_vector< VertexType >& predecessors,
      LoadBalance loadBalance,
      bool withFilter )
   {
      distances.resize( vertexCount );
      predecessors.resize( vertexCount );
      gunrock::options_t options;
      options.advance_load_balance = toGunrockLoadBalance( loadBalance );
      // Without the filter, the invalid vertices stay in the frontier and are skipped by the next advance
      // (Gunrock's default).  With the filter, they are removed by the predicated filter (copy_if).
      options.enable_filter = withFilter;
      gunrock_bfs_with_predecessors::param_t< VertexType > param{ source, options };
      gunrock_bfs_with_predecessors::result_t< VertexType > result{ distances.data().get(), predecessors.data().get() };
      // A fresh context is constructed on every call like in gunrock::bfs::run, since the enactor is not safe to
      // reuse the same multi_context_t across independent runs.
      auto context = std::make_shared< gunrock::gcuda::multi_context_t >( 0 );
      gunrock_bfs_with_predecessors::run( graph, param, result, context );
   }

   /**
    * \brief SSSP from one source (gunrock::sssp::run).
    *
    * The output vectors are resized here, so their allocation is a part of the measured time.
    */
   template< typename Graph >
   static void
   singleSourceShortestPath(
      Graph& graph,
      VertexType source,
      VertexType vertexCount,
      thrust::device_vector< ValueType >& distances,
      thrust::device_vector< VertexType >& predecessors,
      LoadBalance loadBalance )
   {
      distances.resize( vertexCount );
      predecessors.resize( vertexCount );
      gunrock::options_t options;
      options.advance_load_balance = toGunrockLoadBalance( loadBalance );
      gunrock::sssp::param_t< VertexType > param( source, options );
      gunrock::sssp::result_t< VertexType, ValueType > result( distances.data().get(), predecessors.data().get(), vertexCount );
      // See breadthFirstSearch above: a fresh context per call is required.
      auto context = std::make_shared< gunrock::gcuda::multi_context_t >( 0 );
      gunrock::sssp::run( graph, param, result, context );
   }

   //! \brief Releases the memory of a device vector, so that its next resize allocates it again.
   template< typename T >
   static void
   release( thrust::device_vector< T >& vector )
   {
      vector.clear();
      vector.shrink_to_fit();
   }
#endif
};

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#ifdef HAVE_GUNROCK
   #include <gunrock/algorithms/bfs.hxx>
   #include <gunrock/algorithms/sssp.hxx>
#endif

template< typename Real = double, typename Index = int >
struct GunrockBenchmark
{
   using IndexType = Index;
   using RealType = Real;

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
      thrust::copy( d_distances.begin(), d_distances.end(), distances.begin() );
#endif
   }
};

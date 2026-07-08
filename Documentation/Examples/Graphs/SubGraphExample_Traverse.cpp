#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Algorithms/AtomicOperations.h>

template< typename Device >
void
traverseExample()
{
   //! [traverse subgraph]
   using GraphType = TNL::Graphs::Graph< int, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using CounterVector = TNL::Containers::Vector< IndexType, Device, IndexType >;

   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1 }, { 0, 2, 2 },
        { 1, 3, 3 },
        { 2, 3, 4 },
        { 3, 4, 5 },
        { 4, 5, 6 } } );
   // clang-format on

   // SubGraph with vertex filter (exclude vertex 2) and edge filter (w <= 3)
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      },
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );

   // forAllEdges traverses only active vertices and edges that pass the filter
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< Device >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );

   std::cout << "Active edges (vertex 2 excluded, w<=3): " << counter.getElement( 0 ) << "\n";
   //! [traverse subgraph]
}

int
main()
{
   std::cout << "Running on host:\n";
   traverseExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   traverseExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   traverseExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

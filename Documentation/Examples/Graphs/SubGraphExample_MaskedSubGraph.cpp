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
maskedSubGraphExample()
{
   //! [masked subgraph]
   using GraphType = TNL::Graphs::Graph< int, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using IndexVector = TNL::Containers::Vector< IndexType, Device, IndexType >;
   using CounterVector = TNL::Containers::Vector< IndexType, Device, IndexType >;

   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1 }, { 0, 2, 2 },
        { 1, 3, 3 },
        { 2, 3, 4 },
        { 3, 4, 5 },
        { 4, 5, 6 } } );
   // clang-format on

   // Create an MaskedSubGraph from a list of vertex indexes
   // Only vertices {0, 1, 3, 4} are active; vertex 2 and 5 are excluded
   IndexVector indexes{ 0, 1, 3, 4 };
   auto sg = TNL::Graphs::makeSubGraph( graph, indexes );

   std::cout << "Vertex count: " << sg.getVertexCount() << "\n";
   std::cout << "isActive(0): " << sg.isActive( 0 ) << "\n";
   std::cout << "isActive(2): " << sg.isActive( 2 ) << "\n";
   std::cout << "isActive(5): " << sg.isActive( 5 ) << "\n";

   // Traverse: only edges from active vertices are visited
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< Device >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );

   std::cout << "Active edges: " << counter.getElement( 0 ) << "\n";
   //! [masked subgraph]

   //! [masked subgraph edge filter]
   using ValueType = typename GraphType::ValueType;

   // MaskedSubGraph with an additional edge filter (w <= 3)
   auto sg2 = TNL::Graphs::makeSubGraph(
      graph,
      indexes,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );

   CounterVector counter2( 1, 0 );
   auto counter2View = counter2.getView();
   TNL::Graphs::forAllEdges(
      sg2,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< Device >::add( counter2View[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );

   std::cout << "Active edges (masked + w<=3): " << counter2.getElement( 0 ) << "\n";
   //! [masked subgraph edge filter]
}

int
main()
{
   std::cout << "Running on host:\n";
   maskedSubGraphExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   maskedSubGraphExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   maskedSubGraphExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

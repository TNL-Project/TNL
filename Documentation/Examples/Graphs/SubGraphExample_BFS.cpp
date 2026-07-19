#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>

template< typename Device >
void
bfsOnSubgraphExample()
{
   //! [bfs on subgraph]
   using GraphType = TNL::Graphs::Graph< int, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using Vector = TNL::Containers::Vector< IndexType, Device, IndexType >;

   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1 }, { 0, 2, 2 },
        { 1, 3, 3 },
        { 2, 3, 4 },
        { 3, 4, 5 },
        { 4, 5, 6 } } );
   // clang-format on

   // Build a subgraph that excludes vertex 2 (and all edges incident to it)
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );

   // Run BFS on the subgraph — vertices outside the subgraph keep distance -1
   Vector distances( sg.getVertexCount(), -1 );
   TNL::Graphs::Algorithms::breadthFirstSearch( sg, 0, distances );

   std::cout << "Distances from vertex 0 on subgraph (vertex 2 excluded): ";
   for( IndexType i = 0; i < distances.getSize(); i++ )
      std::cout << distances.getElement( i ) << " ";
   std::cout << "\n";
   //! [bfs on subgraph]
}

int
main()
{
   std::cout << "Running on host:\n";
   bfsOnSubgraphExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   bfsOnSubgraphExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   bfsOnSubgraphExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Algorithms/stronglyConnectedComponents.h>

template< typename Device >
void
stronglyConnectedComponentsExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Directed graph with three strongly connected components:
    *
    *    SCC 1:  0 <--> 1         (cycle: 0->1->0)
    *    SCC 2:  2 <--> 3 <--> 4  (cycle: 2->3->4->2)
    *    SCC 3:  5                (single vertex, no self-loop)
    *
    *    Cross edges: 1 -> 2, 4 -> 5  (do not create strong connectivity)
    */
   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1.0f },
        { 1, 0, 1.0f }, { 1, 2, 1.0f },
        { 2, 3, 1.0f },
        { 3, 4, 1.0f },
        { 4, 2, 1.0f }, { 4, 5, 1.0f } } );
   // clang-format on
   std::cout << "Graph:\n" << graph << "\n";

   //! [scc basic]
   /***
    * Basic strongly connected components.
    * Labels start at 1; each vertex gets its SCC label.
    */
   VectorType components;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, components );
   std::cout << "SCC labels: " << components << "\n";
   //! [scc basic]

   //! [scc edge predicate]
   /***
    * Edge-predicate SCC: block the edge (1, 0).
    * This breaks the cycle {0, 1}, so vertices 0 and 1 become separate SCCs.
    */
   auto blockEdge10 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( src == 1 && tgt == 0 );
   };
   VectorType componentsEdge;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, blockEdge10, componentsEdge );
   std::cout << "SCC labels (edge 1->0 blocked): " << componentsEdge << "\n";
   //! [scc edge predicate]

   //! [scc induced]
   /***
    * Induced-subgraph SCC: restrict to vertices {0, 1, 2, 3, 4}.
    * Vertex 5 is inactive and gets label -1.
    */
   VectorType activeVertices{ 0, 1, 2, 3, 4 };
   VectorType componentsInduced;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, activeVertices, componentsInduced );
   std::cout << "SCC labels (induced on {0,1,2,3,4}): " << componentsInduced << "\n";
   //! [scc induced]

   //! [scc induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate SCC.
    */
   VectorType componentsInducedEdge;
   TNL::Graphs::Algorithms::stronglyConnectedComponents( graph, activeVertices, blockEdge10, componentsInducedEdge );
   std::cout << "SCC labels (induced on {0,1,2,3,4}, edge 1->0 blocked): " << componentsInducedEdge << "\n";
   //! [scc induced edge predicate]

   //! [scc if]
   /***
    * Predicate-based SCC: activate only vertices with index < 5.
    */
   auto isActive = [] __cuda_callable__( IndexType vertex )
   {
      return vertex < 5;
   };
   VectorType componentsIf;
   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graph, isActive, componentsIf );
   std::cout << "SCC labels (active if vertex < 5): " << componentsIf << "\n";
   //! [scc if]

   //! [scc if edge predicate]
   /***
    * Combined predicate + edge-predicate SCC.
    */
   VectorType componentsIfEdge;
   TNL::Graphs::Algorithms::stronglyConnectedComponentsIf( graph, isActive, blockEdge10, componentsIfEdge );
   std::cout << "SCC labels (active if vertex < 5, edge 1->0 blocked): " << componentsIfEdge << "\n";
   //! [scc if edge predicate]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   stronglyConnectedComponentsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   stronglyConnectedComponentsExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   stronglyConnectedComponentsExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

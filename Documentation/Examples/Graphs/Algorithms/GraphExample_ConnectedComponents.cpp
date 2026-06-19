#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Algorithms/connectedComponents.h>

template< typename Device >
void
connectedComponentsExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Directed graph with two weakly connected components:
    *
    *    Component 1:  0 ---> 1 ---> 2     (vertices 0, 1, 2)
    *    Component 2:  3 ---> 4            (vertices 3, 4)
    *    Isolated:     5                   (vertex 5, its own component)
    *
    * CC treats the graph as undirected (weakly connected components).
    */
   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1.0f }, { 1, 2, 1.0f },
                       { 3, 4, 1.0f } } );
   // clang-format on
   std::cout << "Graph:\n" << graph << "\n";

   //! [cc basic]
   /***
    * Basic connected components: each vertex is labeled with the smallest
    * vertex index in its component. Isolated vertices form their own
    * component.
    */
   VectorType components;
   TNL::Graphs::Algorithms::connectedComponents( graph, components );
   std::cout << "Components: " << components << "\n";
   //! [cc basic]

   //! [cc edge predicate]
   /***
    * Edge-predicate CC: ignore the edge (1, 2).
    * This splits component {0, 1, 2} into {0, 1} and {2}.
    */
   auto blockEdge12 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( src == 1 && tgt == 2 );
   };
   VectorType componentsEdge;
   TNL::Graphs::Algorithms::connectedComponents( graph, blockEdge12, componentsEdge );
   std::cout << "Components (edge 1->2 blocked): " << componentsEdge << "\n";
   //! [cc edge predicate]

   //! [cc induced]
   /***
    * Induced-subgraph CC: restrict to vertices {0, 1, 3, 4, 5}.
    * Vertex 2 is inactive and gets label -1.
    */
   VectorType activeVertices{ 0, 1, 3, 4, 5 };
   VectorType componentsInduced;
   TNL::Graphs::Algorithms::connectedComponents( graph, activeVertices, componentsInduced );
   std::cout << "Components (induced on {0,1,3,4,5}): " << componentsInduced << "\n";
   //! [cc induced]

   //! [cc induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate CC.
    */
   VectorType componentsInducedEdge;
   TNL::Graphs::Algorithms::connectedComponents( graph, activeVertices, blockEdge12, componentsInducedEdge );
   std::cout << "Components (induced on {0,1,3,4,5}, edge 1->2 blocked): " << componentsInducedEdge << "\n";
   //! [cc induced edge predicate]

   //! [cc if]
   /***
    * Predicate-based CC: activate only vertices with index != 2.
    */
   auto isActive = [] __cuda_callable__( IndexType vertex )
   {
      return vertex != 2;
   };
   VectorType componentsIf;
   TNL::Graphs::Algorithms::connectedComponentsIf( graph, isActive, componentsIf );
   std::cout << "Components (active if vertex != 2): " << componentsIf << "\n";
   //! [cc if]

   //! [cc if edge predicate]
   /***
    * Combined predicate + edge-predicate CC.
    */
   VectorType componentsIfEdge;
   TNL::Graphs::Algorithms::connectedComponentsIf( graph, isActive, blockEdge12, componentsIfEdge );
   std::cout << "Components (active if vertex != 2, edge 1->2 blocked): " << componentsIfEdge << "\n";
   //! [cc if edge predicate]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   connectedComponentsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   connectedComponentsExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   connectedComponentsExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

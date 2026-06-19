#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>

template< typename Device >
void
breadthFirstSearchExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Directed graph used in all examples below:
    *
    *    0 ---> 1 ---> 3 ---> 4 ---> 5
    *     \           ^
    *      \--> 2 ----/
    */
   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1.0 }, { 0, 2, 1.0 },
        { 1, 3, 1.0 },
        { 2, 3, 1.0 },
        { 3, 4, 1.0 },
        { 4, 5, 1.0 } } );
   // clang-format on
   std::cout << "Graph:\n" << graph << "\n";

   //! [bfs basic]
   /***
    * Basic BFS: compute distances from vertex 0.
    * Unreachable vertices keep the value -1.
    */
   VectorType distances;
   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, distances );
   std::cout << "Distances from 0: " << distances << "\n";
   //! [bfs basic]

   //! [bfs edge predicate]
   /***
    * Edge-predicate BFS: ignore edges whose target is vertex 3.
    * The lambda returns false for edges that should not be traversed.
    */
   auto skipTarget3 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return tgt != 3;
   };
   VectorType distancesEdge;
   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, skipTarget3, distancesEdge );
   std::cout << "Distances from 0 (skipping edges to 3): " << distancesEdge << "\n";
   //! [bfs edge predicate]

   //! [bfs induced]
   /***
    * Induced-subgraph BFS: restrict traversal to vertices {0, 1, 2, 3}.
    * Vertices 4 and 5 are inactive and stay at distance -1.
    */
   VectorType activeVertices{ 0, 1, 2, 3 };
   VectorType distancesInduced;
   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, activeVertices, distancesInduced );
   std::cout << "Distances from 0 (induced on {0,1,2,3}): " << distancesInduced << "\n";
   //! [bfs induced]

   //! [bfs induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate BFS.
    */
   VectorType distancesInducedEdge;
   TNL::Graphs::Algorithms::breadthFirstSearch( graph, 0, activeVertices, skipTarget3, distancesInducedEdge );
   std::cout << "Distances from 0 (induced on {0,1,2,3}, skipping edges to 3): " << distancesInducedEdge << "\n";
   //! [bfs induced edge predicate]

   //! [bfs if]
   /***
    * Predicate-based BFS: activate only vertices with index < 4.
    * Equivalent to the induced-subgraph overload but with a generic callable.
    */
   auto isActive = [] __cuda_callable__( IndexType vertex )
   {
      return vertex < 4;
   };
   VectorType distancesIf;
   TNL::Graphs::Algorithms::breadthFirstSearchIf( graph, 0, isActive, distancesIf );
   std::cout << "Distances from 0 (active if vertex < 4): " << distancesIf << "\n";
   //! [bfs if]

   //! [bfs if edge predicate]
   /***
    * Combined predicate + edge-predicate BFS.
    */
   VectorType distancesIfEdge;
   TNL::Graphs::Algorithms::breadthFirstSearchIf( graph, 0, isActive, skipTarget3, distancesIfEdge );
   std::cout << "Distances from 0 (active if vertex < 4, skipping edges to 3): " << distancesIfEdge << "\n";
   //! [bfs if edge predicate]

   //! [bfs visitor]
   /***
    * Visitor BFS: a callable is invoked for every reached vertex as
    * visitor(vertex, distance). Here we record the distance of each visited
    * vertex into a separate vector via a view, which is safe on all devices.
    */
   VectorType distancesVisitor;
   VectorType visitedDistances( graph.getVertexCount(), -1 );
   auto visitedDistancesView = visitedDistances.getView();
   auto visitor = [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
   {
      visitedDistancesView[ vertex ] = distance;
   };
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor( graph, 0, visitor, distancesVisitor );
   std::cout << "Visited distances: " << visitedDistances << "\n";
   //! [bfs visitor]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   breadthFirstSearchExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   breadthFirstSearchExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   breadthFirstSearchExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

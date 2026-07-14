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
    * Edge-predicate BFS: skip edges whose target is vertex 3.
    * The lambda (src, tgt, weight) -> bool returns false for edges
    * that should not be traversed.
    */
   auto skipEdgeTo3 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return tgt != 3;
   };
   VectorType distancesEdge;
   auto sgEdge = TNL::Graphs::makeSubGraph( graph, TNL::Graphs::edgeOnly, skipEdgeTo3 );
   TNL::Graphs::Algorithms::breadthFirstSearch( sgEdge, 0, distancesEdge );
   std::cout << "Distances from 0 (skipping edges to 3): " << distancesEdge << "\n";
   //! [bfs edge predicate]

   //! [bfs induced]
   /***
    * Induced-subgraph BFS: restrict traversal to vertices {0, 1, 2, 3}.
    * Vertices 4 and 5 are inactive and stay at distance -1.
    */
   VectorType distancesInduced;
   auto sgInduced = TNL::Graphs::makeSubGraph( graph, VectorType{ 0, 1, 2, 3 } );
   TNL::Graphs::Algorithms::breadthFirstSearch( sgInduced, 0, distancesInduced );
   std::cout << "Distances from 0 (induced on {0,1,2,3}): " << distancesInduced << "\n";
   //! [bfs induced]

   //! [bfs induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate BFS.
    */
   VectorType distancesInducedEdge;
   auto sgInducedEdge = TNL::Graphs::makeSubGraph( graph, VectorType{ 0, 1, 2, 3 }, skipEdgeTo3 );
   TNL::Graphs::Algorithms::breadthFirstSearch( sgInducedEdge, 0, distancesInducedEdge );
   std::cout << "Distances from 0 (induced on {0,1,2,3}, skipping edges to 3): " << distancesInducedEdge << "\n";
   //! [bfs induced edge predicate]

   //! [bfs if]
   /***
    * Predicate-based BFS: activate only vertices with index < 4.
    * The vertex predicate (vertex) -> bool selects active vertices.
    */
   auto activeLt4 = [] __cuda_callable__( IndexType vertex )
   {
      return vertex < 4;
   };
   VectorType distancesIf;
   auto sgIf = TNL::Graphs::makeSubGraph( graph, activeLt4 );
   TNL::Graphs::Algorithms::breadthFirstSearch( sgIf, 0, distancesIf );
   std::cout << "Distances from 0 (active if vertex < 4): " << distancesIf << "\n";
   //! [bfs if]

   //! [bfs if edge predicate]
   /***
    * Combined predicate + edge-predicate BFS.
    */
   VectorType distancesIfEdge;
   auto sgIfEdge = TNL::Graphs::makeSubGraph( graph, activeLt4, skipEdgeTo3 );
   TNL::Graphs::Algorithms::breadthFirstSearch( sgIfEdge, 0, distancesIfEdge );
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
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      graph,
      0,
      // visitor
      [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
      {
         visitedDistancesView[ vertex ] = distance;
      },
      distancesVisitor );
   std::cout << "Visited distances: " << visitedDistances << "\n";
   //! [bfs visitor]

   //! [bfs visitor induced]
   /***
    * Induced-subgraph visitor BFS: restrict traversal to vertices {0, 1, 2, 3}.
    */
   VectorType distancesVisitorInduced;
   VectorType visitedDistancesInduced( graph.getVertexCount(), -1 );
   auto visitedDistancesInducedView = visitedDistancesInduced.getView();
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      sgInduced,
      0,
      // visitor
      [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
      {
         visitedDistancesInducedView[ vertex ] = distance;
      },
      distancesVisitorInduced );
   std::cout << "Visited distances (induced on {0,1,2,3}): " << visitedDistancesInduced << "\n";
   //! [bfs visitor induced]

   //! [bfs visitor if]
   /***
    * Predicate-based visitor BFS: activate only vertices with index < 4.
    */
   VectorType distancesVisitorIf;
   VectorType visitedDistancesIf( graph.getVertexCount(), -1 );
   auto visitedDistancesIfView = visitedDistancesIf.getView();
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      sgIf,
      0,
      // visitor
      [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
      {
         visitedDistancesIfView[ vertex ] = distance;
      },
      distancesVisitorIf );
   std::cout << "Visited distances (active if vertex < 4): " << visitedDistancesIf << "\n";
   //! [bfs visitor if]

   //! [bfs visitor edge predicate]
   /***
    * Edge-predicate visitor BFS: skip edges targeting vertex 3 and record
    * the distance of each visited vertex via a view-capturing visitor.
    */
   VectorType distancesVisitorEdge;
   VectorType visitedDistancesEdge( graph.getVertexCount(), -1 );
   auto visitedDistancesEdgeView = visitedDistancesEdge.getView();
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      sgEdge,
      0,
      // visitor
      [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
      {
         visitedDistancesEdgeView[ vertex ] = distance;
      },
      distancesVisitorEdge );
   std::cout << "Visited distances (skipping edges to 3): " << visitedDistancesEdge << "\n";
   //! [bfs visitor edge predicate]

   //! [bfs visitor induced edge predicate]
   /***
    * Induced-subgraph + edge-predicate visitor BFS: restrict traversal to
    * vertices {0, 1, 2, 3} and skip edges targeting vertex 3.
    */
   VectorType distancesVisitorInducedEdge;
   VectorType visitedDistancesInducedEdge( graph.getVertexCount(), -1 );
   auto visitedDistancesInducedEdgeView = visitedDistancesInducedEdge.getView();
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      sgInducedEdge,
      0,
      // visitor
      [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
      {
         visitedDistancesInducedEdgeView[ vertex ] = distance;
      },
      distancesVisitorInducedEdge );
   std::cout << "Visited distances (induced on {0,1,2,3}, skipping edges to 3): " << visitedDistancesInducedEdge << "\n";
   //! [bfs visitor induced edge predicate]

   //! [bfs visitor if edge predicate]
   /***
    * Predicate + edge-predicate visitor BFS: activate only vertices with
    * index < 4 and skip edges targeting vertex 3.
    */
   VectorType distancesVisitorIfEdge;
   VectorType visitedDistancesIfEdge( graph.getVertexCount(), -1 );
   auto visitedDistancesIfEdgeView = visitedDistancesIfEdge.getView();
   TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
      sgIfEdge,
      0,
      // visitor
      [ = ] __cuda_callable__( IndexType vertex, IndexType distance ) mutable
      {
         visitedDistancesIfEdgeView[ vertex ] = distance;
      },
      distancesVisitorIfEdge );
   std::cout << "Visited distances (active if vertex < 4, skipping edges to 3): " << visitedDistancesIfEdge << "\n";
   //! [bfs visitor if edge predicate]
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

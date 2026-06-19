#include <iostream>
#include <limits>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Algorithms/singleSourceShortestPath.h>

template< typename Device >
void
singleSourceShortestPathExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using VectorType = TNL::Containers::Vector< ValueType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Weighted directed graph used in all examples below:
    *
    *    0 --(2)--> 1 --(3)--> 3 --(1)--> 4 --(4)--> 5
    *     \                    ^
    *      --(5)--> 2 --(2)---/
    */
   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 2.0f }, { 0, 2, 5.0f },
        { 1, 3, 3.0f },
        { 2, 3, 2.0f },
        { 3, 4, 1.0f },
        { 4, 5, 4.0f } } );
   // clang-format on
   std::cout << "Graph:\n" << graph << "\n";

   //! [sssp basic]
   /***
    * Basic SSSP: shortest-path distances from vertex 0.
    * Unreachable vertices keep the value -1.
    */
   VectorType distances;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, distances );
   std::cout << "Distances from 0: " << distances << "\n";
   //! [sssp basic]

   //! [sssp edge weight callable]
   /***
    * Edge-weight callable SSSP: double the weight of edge (0,1).
    * Returning infinity marks an edge as non-traversable.
    */
   auto transformWeights = [] __cuda_callable__( IndexType src, IndexType tgt, ValueType w ) -> ValueType
   {
      if( src == 0 && tgt == 1 )
         return w * 2;
      return w;
   };
   VectorType distancesEdge;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, transformWeights, distancesEdge );
   std::cout << "Distances from 0 (edge 0->1 doubled): " << distancesEdge << "\n";
   //! [sssp edge weight callable]

   //! [sssp induced]
   /***
    * Induced-subgraph SSSP: restrict to vertices {0, 1, 2, 3}.
    * Vertices 4 and 5 are inactive and stay at distance -1.
    */
   TNL::Containers::Vector< IndexType, Device, IndexType > activeVertices{ 0, 1, 2, 3 };
   VectorType distancesInduced;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, activeVertices, distancesInduced );
   std::cout << "Distances from 0 (induced on {0,1,2,3}): " << distancesInduced << "\n";
   //! [sssp induced]

   //! [sssp induced edge weight callable]
   /***
    * Combined induced-subgraph + edge-weight callable SSSP.
    */
   VectorType distancesInducedEdge;
   TNL::Graphs::Algorithms::singleSourceShortestPath( graph, 0, activeVertices, transformWeights, distancesInducedEdge );
   std::cout << "Distances from 0 (induced on {0,1,2,3}, edge 0->1 doubled): " << distancesInducedEdge << "\n";
   //! [sssp induced edge weight callable]

   //! [sssp if]
   /***
    * Predicate-based SSSP: activate only vertices with index < 4.
    */
   auto isActive = [] __cuda_callable__( IndexType vertex )
   {
      return vertex < 4;
   };
   VectorType distancesIf;
   TNL::Graphs::Algorithms::singleSourceShortestPathIf( graph, 0, isActive, distancesIf );
   std::cout << "Distances from 0 (active if vertex < 4): " << distancesIf << "\n";
   //! [sssp if]

   //! [sssp if edge weight callable]
   /***
    * Combined predicate + edge-weight callable SSSP.
    */
   VectorType distancesIfEdge;
   TNL::Graphs::Algorithms::singleSourceShortestPathIf( graph, 0, isActive, transformWeights, distancesIfEdge );
   std::cout << "Distances from 0 (active if vertex < 4, edge 0->1 doubled): " << distancesIfEdge << "\n";
   //! [sssp if edge weight callable]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   singleSourceShortestPathExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   singleSourceShortestPathExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   singleSourceShortestPathExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

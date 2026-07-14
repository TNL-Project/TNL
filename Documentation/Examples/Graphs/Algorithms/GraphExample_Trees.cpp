#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/Algorithms/trees.h>

template< typename Device >
void
treesExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Tree graph (6 vertices, 5 edges, no cycles):
    *
    *       0
    *      / \
    *     1   2
    *    / \   \
    *   3   4   5
    */
   // clang-format off
   GraphType treeGraph( 6,
      { { 0, 1, 1.0f }, { 0, 2, 1.0f },
          { 1, 3, 1.0f }, { 1, 4, 1.0f },
                         { 2, 5, 1.0f } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on

   /***
    * Cyclic graph (5 vertices, 5 edges, cycle 1-2-3-4-1):
    *
    *   0 --- 1 --- 2
    *         |     |
    *         4 --- 3
    */
   // clang-format off
   GraphType cyclicGraph( 5,
      { { 0, 1, 1.0f },
          { 1, 2, 1.0f }, { 1, 4, 1.0f },
                         { 2, 3, 1.0f },
                                        { 3, 4, 1.0f } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on

   /***
    * Forest graph (5 vertices, 3 edges, two trees):
    *
    *    0 --- 1     2 --- 3 --- 4
    */
   // clang-format off
   GraphType forestGraph( 5,
      { { 0, 1, 1.0f },
          { 2, 3, 1.0f }, { 3, 4, 1.0f } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on

   std::cout << "Tree graph:\n"
             << treeGraph << "\n"
             << "Cyclic graph:\n"
             << cyclicGraph << "\n"
             << "Forest graph:\n"
             << forestGraph << "\n";

   // ===== isTree =====

   //! [is tree basic]
   /***
    * Basic isTree: check if treeGraph is a tree starting from vertex 0.
    */
   bool isTreeResult = TNL::Graphs::Algorithms::isTree( treeGraph, 0 );
   std::cout << "isTree(treeGraph, 0): " << ( isTreeResult ? "true" : "false" ) << "\n";
   //! [is tree basic]

   //! [is tree edge predicate]
   /***
    * Edge-predicate isTree: block edge (0, 2).
    * The lambda (src, tgt, weight) -> bool returns false for blocked edges.
    * This disconnects vertices 2 and 5, so it is not a tree.
    */
   auto blockEdge02 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( ( src == 0 && tgt == 2 ) || ( src == 2 && tgt == 0 ) );
   };
   auto sgTreeEdge = TNL::Graphs::makeSubGraph( treeGraph, TNL::Graphs::edgeOnly, blockEdge02 );
   bool isTreeEdgeResult = TNL::Graphs::Algorithms::isTree( sgTreeEdge, 0 );
   std::cout << "isTree(treeGraph, 0, block 0-2): " << ( isTreeEdgeResult ? "true" : "false" ) << "\n";
   //! [is tree edge predicate]

   //! [is tree induced]
   /***
    * Induced-subgraph isTree: restrict to vertices {0, 1, 2, 3}.
    * The induced subgraph has edges {0-1, 0-2, 1-3} — a tree of 4 vertices.
    */
   auto sgTreeInduced = TNL::Graphs::makeSubGraph( treeGraph, VectorType{ 0, 1, 2, 3 } );
   bool isTreeInducedResult = TNL::Graphs::Algorithms::isTree( sgTreeInduced, 0 );
   std::cout << "isTree(treeGraph, 0, {0,1,2,3}): " << ( isTreeInducedResult ? "true" : "false" ) << "\n";
   //! [is tree induced]

   //! [is tree induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate isTree.
    */
   auto sgTreeInducedEdge = TNL::Graphs::makeSubGraph( treeGraph, VectorType{ 0, 1, 2, 3 }, blockEdge02 );
   bool isTreeInducedEdgeResult = TNL::Graphs::Algorithms::isTree( sgTreeInducedEdge, 0 );
   std::cout << "isTree(treeGraph, 0, {0,1,2,3}, block 0-2): " << ( isTreeInducedEdgeResult ? "true" : "false" ) << "\n";
   //! [is tree induced edge predicate]

   //! [is tree if]
   /***
    * Predicate-based isTree: activate only vertices with index <= 3.
    * The vertex predicate (vertex) -> bool selects active vertices.
    */
   auto activeLe3 = [] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 3;
   };
   auto sgTreeIf = TNL::Graphs::makeSubGraph( treeGraph, activeLe3 );
   bool isTreeIfResult = TNL::Graphs::Algorithms::isTree( sgTreeIf, 0 );
   std::cout << "isTreeIf(treeGraph, 0, vertex <= 3): " << ( isTreeIfResult ? "true" : "false" ) << "\n";
   //! [is tree if]

   //! [is tree if edge predicate]
   /***
    * Combined predicate + edge-predicate isTree.
    */
   auto sgTreeIfEdge = TNL::Graphs::makeSubGraph( treeGraph, activeLe3, blockEdge02 );
   bool isTreeIfEdgeResult = TNL::Graphs::Algorithms::isTree( sgTreeIfEdge, 0 );
   std::cout << "isTreeIf(treeGraph, 0, vertex <= 3, block 0-2): " << ( isTreeIfEdgeResult ? "true" : "false" ) << "\n";
   //! [is tree if edge predicate]

   // ===== isForest =====

   //! [is forest basic]
   /***
    * Basic isForest: check if forestGraph is a forest (auto-detected roots).
    */
   bool isForestResult = TNL::Graphs::Algorithms::isForest( forestGraph );
   std::cout << "isForest(forestGraph): " << ( isForestResult ? "true" : "false" ) << "\n";
   //! [is forest basic]

   //! [is forest edge predicate]
   /***
    * Edge-predicate isForest on the cyclic graph.
    * Blocking edge (4, 1) removes the cycle, making it a path
    * (which is a tree, hence a forest).
    */
   auto blockEdge41 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( ( src == 4 && tgt == 1 ) || ( src == 1 && tgt == 4 ) );
   };
   auto sgForestEdge = TNL::Graphs::makeSubGraph( cyclicGraph, TNL::Graphs::edgeOnly, blockEdge41 );
   bool isForestEdgeResult = TNL::Graphs::Algorithms::isForest( sgForestEdge );
   std::cout << "isForest(cyclicGraph, block 4-1): " << ( isForestEdgeResult ? "true" : "false" ) << "\n";
   //! [is forest edge predicate]

   //! [is forest induced]
   /***
    * Induced-subgraph isForest: restrict forestGraph to {0, 1, 3, 4}.
    * Edges {2-3, 3-4} become {3-4} (vertex 2 removed) — a forest.
    */
   auto sgForestInduced = TNL::Graphs::makeSubGraph( forestGraph, VectorType{ 0, 1, 3, 4 } );
   bool isForestInducedResult = TNL::Graphs::Algorithms::isForest( sgForestInduced );
   std::cout << "isForest(forestGraph, {0,1,3,4}): " << ( isForestInducedResult ? "true" : "false" ) << "\n";
   //! [is forest induced]

   //! [is forest induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate isForest.
    */
   auto sgForestInducedEdge = TNL::Graphs::makeSubGraph( forestGraph, VectorType{ 0, 1, 3, 4 }, blockEdge41 );
   bool isForestInducedEdgeResult = TNL::Graphs::Algorithms::isForest( sgForestInducedEdge );
   std::cout << "isForest(forestGraph, {0,1,3,4}, block 4-1): " << ( isForestInducedEdgeResult ? "true" : "false" ) << "\n";
   //! [is forest induced edge predicate]

   //! [is forest if]
   /***
    * Predicate-based isForest: activate only even vertices.
    */
   auto evenVertices = [] __cuda_callable__( IndexType vertex )
   {
      return vertex % 2 == 0;
   };
   auto sgForestIf = TNL::Graphs::makeSubGraph( forestGraph, evenVertices );
   bool isForestIfResult = TNL::Graphs::Algorithms::isForest( sgForestIf );
   std::cout << "isForestIf(forestGraph, even vertices): " << ( isForestIfResult ? "true" : "false" ) << "\n";
   //! [is forest if]

   //! [is forest if edge predicate]
   /***
    * Combined predicate + edge-predicate isForest.
    */
   auto sgForestIfEdge = TNL::Graphs::makeSubGraph( forestGraph, evenVertices, blockEdge41 );
   bool isForestIfEdgeResult = TNL::Graphs::Algorithms::isForest( sgForestIfEdge );
   std::cout << "isForestIf(forestGraph, even, block 4-1): " << ( isForestIfEdgeResult ? "true" : "false" ) << "\n";
   //! [is forest if edge predicate]

   // ===== isForestWithRoots =====

   //! [is forest with roots basic]
   /***
    * Basic isForestWithRoots: use explicit root candidates {0, 2}.
    * Each root starts a BFS for one tree component.
    */
   bool isForestWithRootsResult = TNL::Graphs::Algorithms::isForestWithRoots( forestGraph, VectorType{ 0, 2 } );
   std::cout << "isForestWithRoots(forestGraph, {0,2}): " << ( isForestWithRootsResult ? "true" : "false" ) << "\n";
   //! [is forest with roots basic]

   //! [is forest with roots edge predicate]
   /***
    * Edge-predicate isForestWithRoots: block edge (3, 4).
    * This splits the tree {2-3-4} into {2-3} and {4}, so vertex 4 must
    * be added as an extra root to cover the whole forest.
    */
   auto blockEdge34 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( ( src == 3 && tgt == 4 ) || ( src == 4 && tgt == 3 ) );
   };
   auto sgRootsEdge = TNL::Graphs::makeSubGraph( forestGraph, TNL::Graphs::edgeOnly, blockEdge34 );
   bool isForestWithRootsEdgeResult = TNL::Graphs::Algorithms::isForestWithRoots( sgRootsEdge, VectorType{ 0, 2, 4 } );
   std::cout << "isForestWithRoots(forestGraph, block 3-4, {0,2,4}): " << ( isForestWithRootsEdgeResult ? "true" : "false" )
             << "\n";
   //! [is forest with roots edge predicate]

   //! [is forest with roots induced]
   /***
    * Induced-subgraph isForestWithRoots: restrict to {0, 1, 3, 4} with roots {0, 3}.
    */
   auto sgRootsInduced = TNL::Graphs::makeSubGraph( forestGraph, VectorType{ 0, 1, 3, 4 } );
   bool isForestWithRootsInducedResult = TNL::Graphs::Algorithms::isForestWithRoots( sgRootsInduced, VectorType{ 0, 3 } );
   std::cout << "isForestWithRoots(forestGraph, {0,1,3,4}, {0,3}): " << ( isForestWithRootsInducedResult ? "true" : "false" )
             << "\n";
   //! [is forest with roots induced]

   //! [is forest with roots induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate isForestWithRoots.
    * Block edge (3, 4) in the induced subgraph on {0, 1, 3, 4}:
    * edges {0-1, 3-4} become {0-1}, so vertex 4 needs its own root.
    */
   auto sgRootsInducedEdge = TNL::Graphs::makeSubGraph( forestGraph, VectorType{ 0, 1, 3, 4 }, blockEdge34 );
   bool isForestWithRootsInducedEdgeResult =
      TNL::Graphs::Algorithms::isForestWithRoots( sgRootsInducedEdge, VectorType{ 0, 3, 4 } );
   std::cout << "isForestWithRoots(forestGraph, {0,1,3,4}, block 3-4, {0,3,4}): "
             << ( isForestWithRootsInducedEdgeResult ? "true" : "false" ) << "\n";
   //! [is forest with roots induced edge predicate]

   //! [is forest with roots if]
   /***
    * Predicate-based isForestWithRoots: activate vertices <= 3 via predicate, roots {0, 3}.
    */
   auto sgRootsIf = TNL::Graphs::makeSubGraph( forestGraph, activeLe3 );
   bool isForestWithRootsIfResult = TNL::Graphs::Algorithms::isForestWithRoots( sgRootsIf, VectorType{ 0, 3 } );
   std::cout << "isForestWithRootsIf(forestGraph, vertex <= 3, {0,3}): " << ( isForestWithRootsIfResult ? "true" : "false" )
             << "\n";
   //! [is forest with roots if]

   //! [is forest with roots if edge predicate]
   /***
    * Combined predicate + edge-predicate isForestWithRoots.
    */
   auto sgRootsIfEdge = TNL::Graphs::makeSubGraph( forestGraph, activeLe3, blockEdge34 );
   bool isForestWithRootsIfEdgeResult = TNL::Graphs::Algorithms::isForestWithRoots( sgRootsIfEdge, VectorType{ 0, 3 } );
   std::cout << "isForestWithRootsIf(forestGraph, vertex <= 3, block 3-4, {0,3}): "
             << ( isForestWithRootsIfEdgeResult ? "true" : "false" ) << "\n";
   //! [is forest with roots if edge predicate]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   treesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   treesExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   treesExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

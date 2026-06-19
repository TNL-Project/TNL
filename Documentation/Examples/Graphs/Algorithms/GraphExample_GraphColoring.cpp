#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Algorithms/graphColoring.h>

template< typename Device >
void
graphColoringExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using ColorType = int;
   using VectorType = TNL::Containers::Vector< ColorType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Undirected graph (a cycle: 0 -- 1 -- 2 -- 3 -- 4 -- 0):
    *
    *    0 --- 1
    *    |     |
    *    4 --- 3 --- 2
    */
   // clang-format off
   GraphType graph( 5,
      { { 0, 1, 1.0f },
        { 1, 2, 1.0f },
        { 2, 3, 1.0f },
        { 3, 4, 1.0f },
        { 4, 0, 1.0f } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
   std::cout << "Graph:\n" << graph << "\n";

   // ===== graphColoring (greedy) =====

   //! [coloring basic]
   /***
    * Basic greedy coloring: assign zero-based color labels.
    * A 5-cycle needs at least 3 colors.
    */
   VectorType colors;
   TNL::Graphs::Algorithms::graphColoring( graph, colors );
   std::cout << "Greedy colors: " << colors << "\n";
   //! [coloring basic]

   //! [coloring edge predicate]
   /***
    * Edge-predicate coloring: ignore the edge between 4 and 0.
    * The remaining path 0-1-2-3-4 needs only 2 colors.
    */
   auto blockEdge40 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( ( src == 4 && tgt == 0 ) || ( src == 0 && tgt == 4 ) );
   };
   VectorType colorsEdge;
   TNL::Graphs::Algorithms::graphColoring( graph, blockEdge40, colorsEdge );
   std::cout << "Greedy colors (edge 4-0 blocked): " << colorsEdge << "\n";
   //! [coloring edge predicate]

   //! [coloring induced]
   /***
    * Induced-subgraph coloring: restrict to vertices {0, 1, 2, 3}.
    * Vertex 4 is inactive and gets color -1.
    */
   VectorType activeVertices{ 0, 1, 2, 3 };
   VectorType colorsInduced;
   TNL::Graphs::Algorithms::graphColoring( graph, activeVertices, colorsInduced );
   std::cout << "Greedy colors (induced on {0,1,2,3}): " << colorsInduced << "\n";
   //! [coloring induced]

   //! [coloring induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate coloring.
    */
   VectorType colorsInducedEdge;
   TNL::Graphs::Algorithms::graphColoring( graph, activeVertices, blockEdge40, colorsInducedEdge );
   std::cout << "Greedy colors (induced on {0,1,2,3}, edge 4-0 blocked): " << colorsInducedEdge << "\n";
   //! [coloring induced edge predicate]

   //! [coloring if]
   /***
    * Predicate-based coloring: activate only vertices with index <= 3.
    */
   auto isActive = [] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 3;
   };
   VectorType colorsIf;
   TNL::Graphs::Algorithms::graphColoringIf( graph, isActive, colorsIf );
   std::cout << "Greedy colors (active if vertex <= 3): " << colorsIf << "\n";
   //! [coloring if]

   //! [coloring if edge predicate]
   /***
    * Combined predicate + edge-predicate coloring.
    */
   VectorType colorsIfEdge;
   TNL::Graphs::Algorithms::graphColoringIf( graph, isActive, blockEdge40, colorsIfEdge );
   std::cout << "Greedy colors (active if vertex <= 3, edge 4-0 blocked): " << colorsIfEdge << "\n";
   //! [coloring if edge predicate]

   // ===== graphColoringLuby (Luby MIS-based) =====

   //! [coloring luby basic]
   /***
    * Luby coloring: each color class is a maximal independent set.
    */
   VectorType colorsLuby;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, colorsLuby );
   std::cout << "Luby colors: " << colorsLuby << "\n";
   //! [coloring luby basic]

   //! [coloring luby edge predicate]
   /***
    * Edge-predicate Luby coloring.
    */
   VectorType colorsLubyEdge;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, blockEdge40, colorsLubyEdge );
   std::cout << "Luby colors (edge 4-0 blocked): " << colorsLubyEdge << "\n";
   //! [coloring luby edge predicate]

   //! [coloring luby induced]
   /***
    * Induced-subgraph Luby coloring.
    */
   VectorType colorsLubyInduced;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, activeVertices, colorsLubyInduced );
   std::cout << "Luby colors (induced on {0,1,2,3}): " << colorsLubyInduced << "\n";
   //! [coloring luby induced]

   //! [coloring luby induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate Luby coloring.
    */
   VectorType colorsLubyInducedEdge;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, activeVertices, blockEdge40, colorsLubyInducedEdge );
   std::cout << "Luby colors (induced on {0,1,2,3}, edge 4-0 blocked): " << colorsLubyInducedEdge << "\n";
   //! [coloring luby induced edge predicate]

   //! [coloring luby if]
   /***
    * Predicate-based Luby coloring.
    */
   VectorType colorsLubyIf;
   TNL::Graphs::Algorithms::graphColoringLubyIf( graph, isActive, colorsLubyIf );
   std::cout << "Luby colors (active if vertex <= 3): " << colorsLubyIf << "\n";
   //! [coloring luby if]

   //! [coloring luby if edge predicate]
   /***
    * Combined predicate + edge-predicate Luby coloring.
    */
   VectorType colorsLubyIfEdge;
   TNL::Graphs::Algorithms::graphColoringLubyIf( graph, isActive, blockEdge40, colorsLubyIfEdge );
   std::cout << "Luby colors (active if vertex <= 3, edge 4-0 blocked): " << colorsLubyIfEdge << "\n";
   //! [coloring luby if edge predicate]

   // ===== isProperlyColored (verifier) =====

   //! [is properly colored basic]
   /***
    * Verify that the greedy coloring is proper (no adjacent vertices share a color).
    */
   bool isProper = TNL::Graphs::Algorithms::isProperlyColored( graph, colors );
   std::cout << "isProperlyColored(graph, greedy colors): " << ( isProper ? "true" : "false" ) << "\n";
   //! [is properly colored basic]

   //! [is properly colored edge predicate]
   /***
    * Verify coloring with the same edge predicate.
    */
   bool isProperEdge = TNL::Graphs::Algorithms::isProperlyColored( graph, blockEdge40, colorsEdge );
   std::cout << "isProperlyColored(graph, block 4-0, colors): " << ( isProperEdge ? "true" : "false" ) << "\n";
   //! [is properly colored edge predicate]

   //! [is properly colored induced]
   /***
    * Verify induced-subgraph coloring.
    */
   bool isProperInduced = TNL::Graphs::Algorithms::isProperlyColored( graph, activeVertices, colorsInduced );
   std::cout << "isProperlyColored(graph, {0,1,2,3}, colors): " << ( isProperInduced ? "true" : "false" ) << "\n";
   //! [is properly colored induced]

   //! [is properly colored induced edge predicate]
   /***
    * Verify induced-subgraph coloring with edge predicate.
    */
   bool isProperInducedEdge =
      TNL::Graphs::Algorithms::isProperlyColored( graph, activeVertices, blockEdge40, colorsInducedEdge );
   std::cout << "isProperlyColored(graph, {0,1,2,3}, block 4-0, colors): " << ( isProperInducedEdge ? "true" : "false" )
             << "\n";
   //! [is properly colored induced edge predicate]

   //! [is properly colored if]
   /***
    * Verify predicate-induced subgraph coloring.
    */
   bool isProperIf = TNL::Graphs::Algorithms::isProperlyColoredIf( graph, isActive, colorsIf );
   std::cout << "isProperlyColoredIf(graph, vertex <= 3, colors): " << ( isProperIf ? "true" : "false" ) << "\n";
   //! [is properly colored if]

   //! [is properly colored if edge predicate]
   /***
    * Verify predicate + edge-predicate coloring.
    */
   bool isProperIfEdge = TNL::Graphs::Algorithms::isProperlyColoredIf( graph, isActive, blockEdge40, colorsIfEdge );
   std::cout << "isProperlyColoredIf(graph, vertex <= 3, block 4-0, colors): " << ( isProperIfEdge ? "true" : "false" ) << "\n";
   //! [is properly colored if edge predicate]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   graphColoringExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   graphColoringExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   graphColoringExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

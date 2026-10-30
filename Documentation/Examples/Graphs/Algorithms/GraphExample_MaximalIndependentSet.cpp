#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/Algorithms/maximalIndependentSet.h>

template< typename Device >
void
maximalIndependentSetExample()
{
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, Device, IndexType >;
   //! [graph type definition]

   /***
    * Undirected graph (a path: 0 -- 1 -- 2 -- 3 -- 4):
    *
    *    0 --- 1 --- 2 --- 3 --- 4
    */
   // clang-format off
   GraphType graph( 5,
      { { 0, 1, 1.0f },
        { 1, 2, 1.0f },
        { 2, 3, 1.0f },
        { 3, 4, 1.0f } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
   std::cout << "Graph:\n" << graph << "\n";

   // ===== maximalIndependentSet (producer) =====

   //! [mis basic]
   /***
    * Basic MIS: find a maximal independent set.
    * Output is a 0/1 mask (1 = vertex in the MIS).
    */
   VectorType independentSet;
   TNL::Graphs::Algorithms::maximalIndependentSet( graph, independentSet );
   std::cout << "MIS mask: " << independentSet << "\n";
   //! [mis basic]

   //! [mis edge predicate]
   /***
    * Edge-predicate MIS: ignore the edge between 2 and 3.
    * The lambda (src, tgt, weight) -> bool returns false for blocked edges.
    * Vertices 2 and 3 may now coexist in the independent set.
    */
   auto blockEdge23 = [] __cuda_callable__( IndexType src, IndexType tgt, float )
   {
      return ! ( ( src == 2 && tgt == 3 ) || ( src == 3 && tgt == 2 ) );
   };
   VectorType independentSetEdge;
   auto subgraphEdge = TNL::Graphs::makeSubGraph( graph, TNL::Graphs::edgeOnly, blockEdge23 );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphEdge, independentSetEdge );
   std::cout << "MIS mask (edge 2-3 blocked): " << independentSetEdge << "\n";
   //! [mis edge predicate]

   //! [mis induced]
   /***
    * Induced-subgraph MIS: restrict to vertices {0, 1, 2, 3}.
    * Vertex 4 is inactive and remains 0 in the output.
    */
   VectorType independentSetInduced;
   auto subgraphInduced = TNL::Graphs::makeSubGraph( graph, VectorType{ 0, 1, 2, 3 } );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphInduced, independentSetInduced );
   std::cout << "MIS mask (induced on {0,1,2,3}): " << independentSetInduced << "\n";
   //! [mis induced]

   //! [mis induced edge predicate]
   /***
    * Combined induced-subgraph + edge-predicate MIS.
    */
   VectorType independentSetInducedEdge;
   auto subgraphInducedEdge = TNL::Graphs::makeSubGraph( graph, VectorType{ 0, 1, 2, 3 }, blockEdge23 );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphInducedEdge, independentSetInducedEdge );
   std::cout << "MIS mask (induced on {0,1,2,3}, edge 2-3 blocked): " << independentSetInducedEdge << "\n";
   //! [mis induced edge predicate]

   //! [mis if]
   /***
    * Predicate-based MIS: activate only vertices with index <= 3.
    * The vertex predicate (vertex) -> bool selects active vertices.
    */
   auto activeLessOrEqual3 = [] __cuda_callable__( IndexType vertex )
   {
      return vertex <= 3;
   };
   VectorType independentSetIf;
   auto subgraphIf = TNL::Graphs::makeSubGraph( graph, activeLessOrEqual3 );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphIf, independentSetIf );
   std::cout << "MIS mask (active if vertex <= 3): " << independentSetIf << "\n";
   //! [mis if]

   //! [mis if edge predicate]
   /***
    * Combined predicate + edge-predicate MIS.
    */
   VectorType independentSetIfEdge;
   auto subgraphIfEdge = TNL::Graphs::makeSubGraph( graph, activeLessOrEqual3, blockEdge23 );
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphIfEdge, independentSetIfEdge );
   std::cout << "MIS mask (active if vertex <= 3, edge 2-3 blocked): " << independentSetIfEdge << "\n";
   //! [mis if edge predicate]

   // ===== isMaximalIndependentSet (verifier) =====

   //! [is mis basic]
   /***
    * Verify that the computed mask is a valid maximal independent set.
    */
   VectorType independentSetForCheck;
   TNL::Graphs::Algorithms::maximalIndependentSet( graph, independentSetForCheck );
   bool isValid = TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, independentSetForCheck );
   std::cout << "isMaximalIndependentSet(graph, mis): " << ( isValid ? "true" : "false" ) << "\n";
   //! [is mis basic]

   //! [is mis edge predicate]
   /***
    * Verify MIS with the same edge predicate used during computation.
    */
   VectorType independentSetForCheckEdge;
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphEdge, independentSetForCheckEdge );
   bool isValidEdge = TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphEdge, independentSetForCheckEdge );
   std::cout << "isMaximalIndependentSet(graph, block 2-3, mis): " << ( isValidEdge ? "true" : "false" ) << "\n";
   //! [is mis edge predicate]

   //! [is mis induced]
   /***
    * Verify MIS on the induced subgraph.
    */
   VectorType independentSetForCheckInduced;
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphInduced, independentSetForCheckInduced );
   bool isValidInduced = TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphInduced, independentSetForCheckInduced );
   std::cout << "isMaximalIndependentSet(graph, {0,1,2,3}, mis): " << ( isValidInduced ? "true" : "false" ) << "\n";
   //! [is mis induced]

   //! [is mis induced edge predicate]
   /***
    * Verify induced-subgraph MIS with edge predicate.
    */
   VectorType independentSetForCheckInducedEdge;
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphInducedEdge, independentSetForCheckInducedEdge );
   bool isValidInducedEdge =
      TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphInducedEdge, independentSetForCheckInducedEdge );
   std::cout << "isMaximalIndependentSet(graph, {0,1,2,3}, block 2-3, mis): " << ( isValidInducedEdge ? "true" : "false" )
             << "\n";
   //! [is mis induced edge predicate]

   //! [is mis if]
   /***
    * Verify predicate-induced subgraph MIS.
    */
   VectorType independentSetForCheckIf;
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphIf, independentSetForCheckIf );
   bool isValidIf = TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphIf, independentSetForCheckIf );
   std::cout << "isMaximalIndependentSetIf(graph, vertex <= 3, mis): " << ( isValidIf ? "true" : "false" ) << "\n";
   //! [is mis if]

   //! [is mis if edge predicate]
   /***
    * Verify predicate + edge-predicate MIS.
    */
   VectorType independentSetForCheckIfEdge;
   TNL::Graphs::Algorithms::maximalIndependentSet( subgraphIfEdge, independentSetForCheckIfEdge );
   bool isValidIfEdge = TNL::Graphs::Algorithms::isMaximalIndependentSet( subgraphIfEdge, independentSetForCheckIfEdge );
   std::cout << "isMaximalIndependentSetIf(graph, vertex <= 3, block 2-3, mis): " << ( isValidIfEdge ? "true" : "false" )
             << "\n";
   //! [is mis if edge predicate]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   maximalIndependentSetExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   maximalIndependentSetExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   maximalIndependentSetExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

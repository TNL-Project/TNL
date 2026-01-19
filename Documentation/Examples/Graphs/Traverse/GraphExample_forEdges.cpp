#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forAllEdgesExample()
{
   /***
    * Create a directed graph with 5 vertices.
    */
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
   // clang-format off
   GraphType graph( 5, // number of vertices
        {  // definition of edges with weights
          { 0, 1, 10.0 }, { 0, 2, 20.0 },
                          { 1, 2, 30.0 }, { 1, 3, 40.0 }, { 1, 4, 50.0 },
                                          { 2, 3, 60.0 },
          { 3, 0, 70.0 },                                { 3, 4, 80.0 } } );
   // clang-format on

   /***
    * Print the graph.
    */
   std::cout << "Graph:\n" << graph << std::endl;

   //! [traverse edges in range]
   /***
    * Traverse edges in range [1, 4) and modify them.
    */
   auto modifyEdge = [] __cuda_callable__( int sourceIdx, int localIdx, int& targetIdx, float& weight ) mutable
   {
      targetIdx = ( targetIdx + 1 ) % 5;
      weight += 5;
   };
   TNL::Graphs::forEdges( graph, 1, 4, modifyEdge );
   //! [traverse edges in range]

   /***
    * Print the modified graph.
    */
   std::cout << "Modified graph:\n" << graph << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   forAllEdgesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   forAllEdgesExample< TNL::Devices::Cuda >();
#endif

   return EXIT_SUCCESS;
}

#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forVerticesExample()
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
          { 3, 0, 70.0 },                                 { 3, 4, 80.0 } } );
   // clang-format on

   /***
    * Print the graph.
    */
   std::cout << "Graph:\n" << graph << '\n';

   //! [traverse vertices in range]
   /***
    * Traverse vertices in range [1, 4) and modify their edges.
    */
   auto processVertex = [] __cuda_callable__( typename GraphType::VertexView vertex ) mutable
   {
      for( int i = 0; i < vertex.getDegree(); i++ )
         vertex.setEdge( i, ( vertex.getTargetIndex( i ) + 1 ) % 5, vertex.getEdgeWeight( i ) + 5 );
   };
   TNL::Graphs::forVertices( graph, 1, 4, processVertex );
   //! [traverse vertices in range]

   /***
    * Print the modified graph.
    */
   std::cout << "Modified graph:\n" << graph << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   forVerticesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   forVerticesExample< TNL::Devices::Cuda >();
#endif

   return EXIT_SUCCESS;
}

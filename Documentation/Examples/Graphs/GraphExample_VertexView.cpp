#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>

template< typename Device >
void
vertexViewExample()
{
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;

   /***
    * Create a directed graph
    */
   // clang-format off
   GraphType graph( 5,
        {  { 0, 1, 10.0 }, { 0, 2, 20.0 },
                           { 1, 2, 30.0 }, { 1, 3, 40.0 }, { 1, 4, 50.0 },
                                           { 2, 3, 60.0 },
           { 3, 0, 70.0 },                                 { 3, 4, 80.0 },
           { 4, 0, 90.0 } } );
   // clang-format on

   std::cout << "Graph:\n" << graph << std::endl;

   /***
    * Modifying edge weights using forAllVertices
    */
   std::cout << "\nExample 3: Modifying edge weights" << std::endl;

   TNL::Graphs::forAllVertices( graph,
                                [] __cuda_callable__( typename GraphType::VertexView vertex )
                                {
                                   for( int i = 0; i < vertex.getDegree(); i++ ) {
                                      vertex.setEdgeWeight( i, vertex.getEdgeWeight( i ) + 1.0 );
                                   }
                                } );

   std::cout << "Graph after modifying edge weights:\n" << graph << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   vertexViewExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   vertexViewExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "Running on HIP device:" << std::endl;
   vertexViewExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

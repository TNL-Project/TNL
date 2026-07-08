#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>

template< typename Device >
void
vertexFilterExample()
{
   //! [vertex filter]
   using GraphType = TNL::Graphs::Graph< int, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;

   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1 }, { 0, 2, 2 },
        { 1, 3, 3 },
        { 2, 3, 4 },
        { 3, 4, 5 },
        { 4, 5, 6 } } );
   // clang-format on

   // Create a subgraph that excludes vertex 2
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );

   std::cout << "Vertex count: " << sg.getVertexCount() << "\n";
   std::cout << "isActive(0): " << sg.isActive( 0 ) << "\n";
   std::cout << "isActive(2): " << sg.isActive( 2 ) << "\n";
   std::cout << "isActive(3): " << sg.isActive( 3 ) << "\n";
   //! [vertex filter]
}

int
main()
{
   std::cout << "Running on host:\n";
   vertexFilterExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   vertexFilterExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   vertexFilterExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

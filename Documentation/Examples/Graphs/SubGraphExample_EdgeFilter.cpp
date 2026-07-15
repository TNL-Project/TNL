#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/SubGraph.h>

template< typename Device >
void
edgeFilterExample()
{
   //! [edge filter]
   using GraphType = TNL::Graphs::Graph< int, Device, int, TNL::Graphs::DirectedGraph >;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

   // clang-format off
   GraphType graph( 6,
      { { 0, 1, 1 }, { 0, 2, 2 },
        { 1, 3, 3 },
        { 2, 3, 4 },
        { 3, 4, 5 },
        { 4, 5, 6 } } );
   // clang-format on

   // Create a subgraph that only keeps edges with weight <= 3
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );

   std::cout << "edgeExists(0,1,1): " << sg.edgeExists( 0, 1, 1 ) << "\n";
   std::cout << "edgeExists(0,2,2): " << sg.edgeExists( 0, 2, 2 ) << "\n";
   std::cout << "edgeExists(2,3,4): " << sg.edgeExists( 2, 3, 4 ) << "\n";
   std::cout << "edgeExists(3,4,5): " << sg.edgeExists( 3, 4, 5 ) << "\n";
   //! [edge filter]

   //! [both filters]
   // Create a subgraph with both vertex and edge filters
   auto sg2 = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      },
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );

   std::cout << "Both filters - vertexExists(2): " << sg2.vertexExists( 2 ) << "\n";
   std::cout << "Both filters - edgeExists(2,3,4): " << sg2.edgeExists( 2, 3, 4 ) << "\n";
   std::cout << "Both filters - edgeExists(0,1,1): " << sg2.edgeExists( 0, 1, 1 ) << "\n";
   //! [both filters]
}

int
main()
{
   std::cout << "Running on host:\n";
   edgeFilterExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "\nRunning on CUDA device:\n";
   edgeFilterExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "\nRunning on HIP device:\n";
   edgeFilterExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

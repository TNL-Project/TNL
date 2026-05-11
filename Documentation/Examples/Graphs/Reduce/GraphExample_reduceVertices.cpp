#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/reduce.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
reduceVerticesExample()
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

   /***
    * Compute maximum edge weight for vertices in range [1, 4).
    */
   //! [vector for results]
   TNL::Containers::Vector< float, Device > vertexMaxWeights( 5, -1 );
   auto vertexMaxWeights_view = vertexMaxWeights.getView();
   //! [vector for results]

   //! [fetch lambda]
   auto fetch = [] __cuda_callable__( int sourceIdx, int targetIdx, const float& weight ) -> float
   {
      return weight;
   };
   //! [fetch lambda]

   //! [store lambda]
   auto store = [ = ] __cuda_callable__( int vertexIdx, const float& maxWeight ) mutable
   {
      vertexMaxWeights_view[ vertexIdx ] = maxWeight;
   };
   //! [store lambda]

   //! [reduce vertices]
   TNL::Graphs::reduceVertices( graph, 1, 4, fetch, TNL::Max{}, store );
   //! [reduce vertices]

   /***
    * Print results.
    */
   std::cout << "Maximum edge weight for vertices 1-3:" << vertexMaxWeights << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   reduceVerticesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   reduceVerticesExample< TNL::Devices::Cuda >();
#endif

   return EXIT_SUCCESS;
}

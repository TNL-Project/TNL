#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/reduce.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
reduceVerticesIfExample()
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
   std::cout << "Graph:\n" << graph << std::endl;

   //! [reduce vertices if]
   /***
    * Compute minimum edge weight for vertices in range [1, 4) with degree >= 2.
    */
   TNL::Containers::Vector< float, Device > vertexMinWeights( 5, -1 ), compressedVertexMinWeights( 5 );
   auto vertexMinWeights_view = vertexMinWeights.getView();
   auto compressedVertexMinWeights_view = compressedVertexMinWeights.getView();

   auto condition = [ = ] __cuda_callable__( int vertexIdx ) -> bool
   {
      return graph.getVertexDegree( vertexIdx ) >= 2;
   };

   auto fetch = [] __cuda_callable__( int sourceIdx, int targetIdx, const float& weight ) -> float
   {
      return weight;
   };

   auto store = [ = ] __cuda_callable__( int indexOfVertexIdx, int vertexIdx, const float& minWeight ) mutable
   {
      compressedVertexMinWeights_view[ indexOfVertexIdx ] = minWeight;
      vertexMinWeights_view[ vertexIdx ] = minWeight;
   };

   int reducedVertexCount = TNL::Graphs::reduceVerticesIf( graph, 1, 4, condition, fetch, TNL::Min{}, store );
   //! [reduce vertices if]

   /***
    * Print results.
    */
   std::cout << "Number of reduced vertices: " << reducedVertexCount << std::endl;
   std::cout << "Minimum edge weight for vertices 1-3 with degree >= 2:" << vertexMinWeights << std::endl;
   std::cout << "Compressed minimum weights:" << compressedVertexMinWeights.getView( 0, reducedVertexCount ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceVerticesIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   reduceVerticesIfExample< TNL::Devices::Cuda >();
#endif

   return EXIT_SUCCESS;
}

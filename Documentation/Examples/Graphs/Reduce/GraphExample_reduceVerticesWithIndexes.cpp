#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/reduce.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
reduceVerticesWithIndexesExample()
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
   std::cout << "Graph:\n" << graph << '\n';

   //! [reduce vertices with indecis]
   /***
    * Compute sum of edge weights for specific vertices (0, 2, 3).
    */
   TNL::Containers::Vector< int, Device > vertexIndices( { 0, 2, 3 } );
   TNL::Containers::Vector< float, Device > vertexSums( 5, -1 );
   TNL::Containers::Vector< float, Device > compressedVertexSums( 3, -1 );
   auto vertexSums_view = vertexSums.getView();
   auto compressedVertexSums_view = compressedVertexSums.getView();

   auto fetch = [] __cuda_callable__( int sourceIdx, int targetIdx, const float& weight ) -> float
   {
      return weight;
   };

   auto store = [ = ] __cuda_callable__( int indexOfVertexIdx, int vertexIdx, const float& sum ) mutable
   {
      compressedVertexSums_view[ indexOfVertexIdx ] = sum;
      vertexSums_view[ vertexIdx ] = sum;
   };

   TNL::Graphs::reduceVertices( graph, vertexIndices, fetch, TNL::Plus{}, store );
   //! [reduce vertices with indecis]

   /***
    * Print results.
    */
   std::cout << "Sum of edge weights for specific vertices:" << vertexSums << '\n';
   std::cout << "Compressed sums:" << compressedVertexSums.getView( 0, vertexIndices.getSize() ) << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   reduceVerticesWithIndexesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   reduceVerticesWithIndexesExample< TNL::Devices::Cuda >();
#endif

   return EXIT_SUCCESS;
}

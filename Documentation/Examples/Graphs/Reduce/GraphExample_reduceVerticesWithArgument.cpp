#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/reduce.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
reduceVerticesWithArgumentExample()
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

   /***
    * Find minimum edge weight and target vertex for vertices in range [1, 4).
    */
   //! [vectors for results]
   TNL::Containers::Vector< float, Device > minWeights( 5, -1 );
   TNL::Containers::Vector< int, Device > minTargets( 5, -1 );
   auto minWeights_view = minWeights.getView();
   auto minTargets_view = minTargets.getView();
   //! [vectors for results]

   //! [fetch lambda]
   auto fetch = [] __cuda_callable__( int sourceIdx, int targetIdx, const float& weight ) -> float
   {
      return weight;
   };
   //! [fetch lambda]

   //! [store lambda]
   auto store = [ = ] __cuda_callable__( int vertexIdx, int localIdx, int targetIdx, float result, bool isolatedVertex ) mutable
   {
      minWeights_view[ vertexIdx ] = result;
      if( ! isolatedVertex )
         minTargets_view[ vertexIdx ] = targetIdx;
   };
   //! [store lambda]

   //! [reduce vertices with argument]
   TNL::Graphs::reduceVerticesWithArgument( graph, 1, 5, fetch, TNL::MinWithArg{}, store );
   //! [reduce vertices with argument]

   /***
    * Print results.
    */
   std::cout << "Minimum edge weight for vertices 1-4:" << minWeights << std::endl;
   std::cout << "Target vertex for minimum edge:" << minTargets << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceVerticesWithArgumentExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   reduceVerticesWithArgumentExample< TNL::Devices::Cuda >();
#endif

   return EXIT_SUCCESS;
}

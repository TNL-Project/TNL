#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>

template< typename Device >
void
setEdgesExample()
{
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;

   //! [setEdges with initializer list]
   /***
    * Example 1: setEdges with initializer list (sparse graph)
    */
   std::cout << "Example 1: setEdges with initializer list (sparse graph)" << std::endl;
   GraphType graph1;
   graph1.setVertexCount( 5 );
   graph1.setEdgeCounts( TNL::Containers::Vector< int, Device >( { 2, 3, 1, 2, 0 } ) );

   // clang-format off
   graph1.setEdges( {
                      { 0, 1, 10.0 }, { 0, 2, 20.0 },
                                      { 1, 2, 30.0 }, { 1, 3, 40.0 }, { 1, 4, 50.0 },
                                                      { 2, 3, 60.0 },
      { 3, 0, 70.0 },                                                 { 3, 4, 80.0 } } );
   // clang-format on

   std::cout << "Sparse graph after setEdges:\n" << graph1 << std::endl;
   //! [setEdges with initializer list]

   //! [setDenseEdges with initializer list]
   /***
    * Example 1b: setEdges with initializer list (dense adjacency matrix)
    * For dense matrix graphs, use nested initializer lists {{row1}, {row2}, ...}
    */
   std::cout << "\nExample 1b: setEdges with initializer list (dense adjacency matrix)" << std::endl;
   using DenseGraphType = TNL::Graphs::Graph< float,
                                              Device,
                                              int,
                                              TNL::Graphs::DirectedGraph,
                                              TNL::Algorithms::Segments::CSR,  // Type of segments is ignored for dense matrices
                                              TNL::Matrices::DenseMatrix< float, Device, int > >;
   DenseGraphType graph1b;
   graph1b.setVertexCount( 4 );

   // clang-format off
   graph1b.setDenseEdges( { {  0.0, 10.0, 20.0,  0.0,  0.0 },  // edges from vertex 0
                            {  0.0,  0.0, 30.0, 40.0, 50.0 },  // edges from vertex 1
                            {  0.0,  0.0,  0.0, 60.0,  0.0 },  // edges from vertex 2
                            { 70.0,  0.0,  0.0,  0.0, 80.0 },  // edges from vertex 3
                            {  0.0,  0.0,  0.0,  0.0,  0.0 } } ); // edges from vertex 4
   // clang-format on

   std::cout << "Dense graph after setDenseEdges:\n" << graph1b << std::endl;
   //! [setDenseEdges with initializer list]

   //! [setEdges with std map]
   /***
    * Example 2: setEdges with std::map
    */
   std::cout << "Example 2: setEdges with std::map" << std::endl;
   GraphType graph2;
   graph2.setVertexCount( 4 );
   graph2.setEdgeCounts( TNL::Containers::Vector< int, Device >( { 2, 2, 1, 1 } ) );

   std::map< std::pair< int, int >, float > edgeMap;
   edgeMap[ { 0, 1 } ] = 1.5;
   edgeMap[ { 0, 2 } ] = 2.5;
   edgeMap[ { 1, 2 } ] = 3.5;
   edgeMap[ { 1, 3 } ] = 4.5;
   edgeMap[ { 2, 3 } ] = 5.5;
   edgeMap[ { 3, 0 } ] = 6.5;

   graph2.setEdges( edgeMap );
   std::cout << "Graph from map:\n" << graph2 << std::endl;
   //! [setEdges with std map]

   /***
    * Example 3: Updating edges
    */
   std::cout << "Example 3: Updating edges" << std::endl;
   // clang-format off
   GraphType graph3( 4, { { 0, 1, 1.0 },
                                         { 1, 2, 2.0 },
                                                       { 2, 3, 3.0 } } );
   // clang-format on

   std::cout << "Original graph:\n" << graph3 << std::endl;

   // Update with new edges (preserving the structure)
   // clang-format off
   graph3.setEdges( { { 0, 1, 10.0 },
                                     { 1, 2, 20.0 },
                                                     { 2, 3, 30.0 } } );
   // clang-format on
   std::cout << "Updated graph:\n" << graph3 << std::endl;

   /***
    * Example 4: setEdges for undirected graph
    */
   std::cout << "Example 4: setEdges for undirected graph" << std::endl;
   using UndirectedGraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;
   UndirectedGraphType graph4;
   graph4.setVertexCount( 4 );
   graph4.setEdgeCounts( TNL::Containers::Vector< int, Device >( { 2, 2, 2, 2 } ) );

   // For undirected graphs, each edge is stored in both directions
   // clang-format off
   graph4.setEdges( {
                     { 0, 1, 1.0 }, { 0, 2, 2.0 },
                                    { 1, 2, 3.0 },
                                                   { 2, 3, 4.0 } }, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   setEdgesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   setEdgesExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "Running on HIP device:" << std::endl;
   setEdgesExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

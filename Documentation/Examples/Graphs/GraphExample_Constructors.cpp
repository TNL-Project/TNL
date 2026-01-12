#include <iostream>
#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
constructorsExample()
{
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;

   /***
    * Example 1: Default constructor
    */
   std::cout << "Example 1: Default constructor" << std::endl;
   GraphType graph1;
   std::cout << "Empty graph - vertices: " << graph1.getVertexCount() << ", edges: " << graph1.getEdgeCount() << "\n"
             << std::endl;

   /***
    * Example 2: Constructor with number of vertices
    */
   std::cout << "Example 2: Constructor with vertex count" << std::endl;
   GraphType graph2( 5 );  // 5 vertices, no edges
   std::cout << "Graph with 5 vertices - vertices: " << graph2.getVertexCount() << ", edges: " << graph2.getEdgeCount() << "\n"
             << std::endl;

   /***
    * Example 3: Constructor with vertices and edges (initializer list - sparse graph)
    */
   std::cout << "Example 3: Constructor with initializer list (sparse graph)" << std::endl;
   // clang-format off
   GraphType graph3( 5, // number of vertices
        {  // definition of edges with weights
                          { 0, 1, 10.0 }, { 0, 2, 20.0 },
                                          { 1, 2, 30.0 }, { 1, 3, 40.0 },
                                                          { 2, 3, 50.0 },
          { 3, 0, 60.0 },                                                 { 3, 4, 70.0 } } );
   // clang-format on
   std::cout << "Sparse graph:\n" << graph3 << std::endl;

   /***
    * Example 3b: Constructor with initializer list only for dense adjacency matrix.
    * For dense matrix graphs, one can also use nested initializer lists {{row1}, {row2}, ...}
    */
   std::cout << "\nExample 3b: Constructor with initializer list (dense adjacency matrix)" << std::endl;
   using DenseGraphType = TNL::Graphs::Graph< float,
                                              Device,
                                              int,
                                              TNL::Graphs::DirectedGraph,
                                              TNL::Algorithms::Segments::CSR,  // Type of segments is ignored for dense matrices
                                              TNL::Matrices::DenseMatrix< float, Device, int > >;
   // clang-format off
   DenseGraphType graph3b( { { 0.0, 10.0, 20.0,  0.0,  0.0 },  // edges from vertex 0
                             { 0.0,  0.0, 30.0, 40.0,  0.0 },  // edges from vertex 1
                             { 0.0,  0.0,  0.0, 50.0,  0.0 },  // edges from vertex 2
                             {60.0,  0.0,  0.0,  0.0, 70.0 },  // edges from vertex 3
                             { 0.0,  0.0,  0.0,  0.0,  0.0 } } ); // edges from vertex 4
   // clang-format on
   std::cout << "Dense graph (complete graph with weighted edges):\n" << graph3b << std::endl;

   /***
    * Example 4: Constructor with vertices and edges (map)
    */
   std::cout << "Example 4: Constructor with std::map" << std::endl;
   std::map< std::pair< int, int >, float > edgeMap;
   edgeMap[ { 0, 1 } ] = 1.5;
   edgeMap[ { 0, 2 } ] = 2.5;
   edgeMap[ { 1, 2 } ] = 3.5;
   edgeMap[ { 2, 3 } ] = 4.5;

   GraphType graph4( 4, edgeMap );
   std::cout << "Graph from map:\n" << graph4 << std::endl;

   /***
    * Example 5: Constructor for undirected graph
    */
   std::cout << "Example 5: Undirected graph constructor" << std::endl;
   using UndirectedGraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;
   // clang-format off
   UndirectedGraphType graph5( 4,
        {  // only need to specify edges once for undirected graph
          { 0, 1, 1.0 }, { 0, 2, 2.0 },
                         { 1, 2, 3.0 },
                                        { 2, 3, 4.0 } }, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
   std::cout << "Undirected graph:\n" << graph5 << std::endl;

   /***
    * Example 6: Copy constructor
    */
   std::cout << "Example 6: Copy constructor" << std::endl;
   GraphType graph6( graph3 );
   std::cout << "Copied graph - vertices: " << graph6.getVertexCount() << ", edges: " << graph6.getEdgeCount() << "\n"
             << std::endl;

   /***
    * Example 7: Constructor from adjacency matrix
    */
   std::cout << "Example 7: Constructor from adjacency matrix" << std::endl;
   using MatrixType = typename GraphType::AdjacencyMatrixType;
   MatrixType matrix( 3, 3 );
   // clang-format off
   matrix.setElements( {                { 0, 1, 1.0 }, { 0, 2, 2.0 },
                                                       { 1, 2, 3.0 },
                         { 2, 0, 4.0 } } );
   // clang-format on

   GraphType graph7( matrix );
   std::cout << "Graph from adjacency matrix:\n" << graph7 << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   constructorsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   constructorsExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "Running on HIP device:" << std::endl;
   constructorsExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

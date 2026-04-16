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
   //! [graph type definition]
   using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
   //! [graph type definition]

   //! [undirected graph type definition]
   using UndirectedGraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;
   //! [undirected graph type definition]

   //! [dense graph type definition]
   using DenseGraphType = TNL::Graphs::Graph< float,
                                              Device,
                                              int,
                                              TNL::Graphs::DirectedGraph,
                                              TNL::Algorithms::Segments::CSR,  // Type of segments is ignored for dense matrices
                                              TNL::Matrices::DenseMatrix< float, Device, int > >;
   //! [dense graph type definition]

   //! [default constructor]
   /***
    * Example 1: Default constructor
    */
   std::cout << "Example 1: Default constructor\n";
   GraphType graph1;
   std::cout << "Empty graph - vertices: " << graph1.getVertexCount() << ", edges: " << graph1.getEdgeCount() << "\n\n";
   //! [default constructor]

   //! [constructor with vertex count]
   /***
    * Example 2: Constructor with number of vertices
    */
   std::cout << "Example 2: Constructor with vertex count\n";
   GraphType graph2( 5 );  // 5 vertices, no edges
   std::cout << "Graph with 5 vertices - vertices: " << graph2.getVertexCount() << ", edges: " << graph2.getEdgeCount() << "\n\n";
   //! [constructor with vertex count]

   //! [constructor with edges]
   /***
    * Example 3: Constructor with vertices and edges (initializer list - sparse graph)
    */
   std::cout << "Example 3a: Constructor with initializer list (sparse graph)\n";
   // clang-format off
   GraphType graph3a( 5, // number of vertices
        {  // definition of edges with weights
                          { 0, 1, 10.0 }, { 0, 2, 20.0 },
                                          { 1, 2, 30.0 }, { 1, 3, 40.0 },
                                                          { 2, 3, 50.0 },
          { 3, 0, 60.0 },                                                 { 3, 4, 70.0 } } );
   // clang-format on
   std::cout << "Sparse graph:\n" << graph3a << '\n';
   //! [constructor with edges]

   //! [constructor with edges for undirected graph]
   /***
    * Example 3b: Constructor for undirected graph
    */
   std::cout << "Example 3b: Undirected graph constructor\n";
   // clang-format off
   UndirectedGraphType graph3b( 4,
        {  // only need to specify edges once for undirected graph
          { 0, 1, 1.0 }, { 0, 2, 2.0 },
                         { 1, 2, 3.0 },
                                        { 2, 3, 4.0 } }, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
   std::cout << "Undirected graph:\n" << graph3b << '\n';
   //! [constructor with edges for undirected graph]

   //! [constructor with edges for dense graph]
   /***
    * Example 3c: Constructor with initializer list only for dense adjacency matrix.
    * For dense matrix graphs, one can also use nested initializer lists {{row1}, {row2}, ...}
    */
   std::cout << "\nExample 3c: Constructor with initializer list (dense adjacency matrix)\n";
   // clang-format off
   DenseGraphType graph3c( { { 0.0, 10.0, 20.0,  0.0,  0.0 },  // edges from vertex 0
                             { 0.0,  0.0, 30.0, 40.0,  0.0 },  // edges from vertex 1
                             { 0.0,  0.0,  0.0, 50.0,  0.0 },  // edges from vertex 2
                             {60.0,  0.0,  0.0,  0.0, 70.0 },  // edges from vertex 3
                             { 0.0,  0.0,  0.0,  0.0,  0.0 } } ); // edges from vertex 4
   // clang-format on
   std::cout << "Dense graph (complete graph with weighted edges):\n" << graph3c << '\n';
   //! [constructor with edges for dense graph]

   //! [constructor with edge map]
   /***
    * Example 4: Constructor with vertices and edges (map)
    */
   std::cout << "\nExample 4: Constructor with std::map\n";
   std::map< std::pair< int, int >, float > edgeMap;
   edgeMap[ { 0, 1 } ] = 1.5;
   edgeMap[ { 0, 2 } ] = 2.5;
   edgeMap[ { 1, 2 } ] = 3.5;
   edgeMap[ { 2, 3 } ] = 4.5;

   GraphType graph4( 4, edgeMap );
   std::cout << "Graph from map:\n" << graph4 << '\n';
   //! [constructor with edge map]

   //! [copy constructor]
   /***
    * Example 5: Copy constructor
    */
   std::cout << "Example 5: Copy constructor\n";
   GraphType graph5( graph3a );
   std::cout << "Copied graph - vertices: " << graph5.getVertexCount() << ", edges: " << graph5.getEdgeCount() << "\n\n";
   //! [copy constructor]

   //! [constructor from adjacency matrix]
   /***
    * Example 6: Constructor from adjacency matrix
    */
   std::cout << "Example 6: Constructor from adjacency matrix\n";
   using MatrixType = typename GraphType::AdjacencyMatrixType;
   MatrixType matrix( 3, 3 );
   // clang-format off
   matrix.setElements( {                { 0, 1, 1.0 }, { 0, 2, 2.0 },
                                                       { 1, 2, 3.0 },
                         { 2, 0, 4.0 } } );
   // clang-format on
   //! [constructor from adjacency matrix]

   GraphType graph6( matrix );
   std::cout << "Graph from adjacency matrix:\n" << graph6 << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   constructorsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   constructorsExample< TNL::Devices::Cuda >();
#endif

#ifdef __HIP__
   std::cout << "Running on HIP device:\n";
   constructorsExample< TNL::Devices::Hip >();
#endif

   return EXIT_SUCCESS;
}

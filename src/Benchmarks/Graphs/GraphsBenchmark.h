// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Algorithms/Graphs/GraphReader.h>
#include <TNL/Algorithms/Graphs/breadthFirstSearch.h>
#include <TNL/Algorithms/Graphs/singleSourceShortestPath.h>
#include "BoostGraph.h"
#include "GunrockBenchmark.h"

template< typename Real = double,
          typename Index = int >
struct GraphsBenchmark
{
   using HostMatrix = TNL::Matrices::SparseMatrix<Real, TNL::Devices::Host, Index>;
   using HostIndexVector = TNL::Containers::Vector<Index, TNL::Devices::Host, Index>;
   using HostRealVector = TNL::Containers::Vector<Real, TNL::Devices::Host, Index>;

   static void configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("input-file", "Input file with the graph." );
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-graphs.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");

      config.addDelimiter( "Device settings:" );
      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "cuda" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "host" );
      config.addEntryEnum< TNL::String >( "sequential" );
      config.addEntryEnum< TNL::String >("cuda");
      TNL::Devices::Host::configSetup( config );
      TNL::Devices::Cuda::configSetup( config );

      config.addEntry<int>("loops", "Number of iterations for every computation.", 10);
      config.addEntry<int>("verbose", "Verbose mode.", 1);
   }

   template< typename Device >
   void TNLBenchmarks( const HostMatrix& hostAdjacencyMatrix, TNL::Benchmarks::Benchmark<>& benchmark, const TNL::String& device )
   {
      using Matrix = TNL::Matrices::SparseMatrix<Real, Device, Index, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRVector >;
      using IndexVector = TNL::Containers::Vector<Index, Device, Index>;
      using RealVector = TNL::Containers::Vector<Real, Device, Index>;

      Matrix adjacencyMatrix;
      adjacencyMatrix = hostAdjacencyMatrix;
      IndexVector bfsDistances( adjacencyMatrix.getRows() );

      // Benchmarking breadth-first search
      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                          { "device", device },
                                                          { "algorithm", std::string( "BFS TNL" ) } } ) );

      auto bfs_tnl = [&] () mutable {
         TNL::Algorithms::Graphs::breadthFirstSearch( adjacencyMatrix, 0, bfsDistances );
      };
      benchmark.time< Device >( device, bfs_tnl );
      if( bfsDistances != this->boostBfsDistances )
      {
         std::cout << "ERROR: Distances do not match!" << std::endl;
         this->errors++;
      }

      // Benchmarking single-source shortest paths
      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType<Real>() },
                                                          { "device", device },
                                                          { "algorithm", std::string( "SSSP TNL" ) } } ) );

      RealVector ssspDistances( adjacencyMatrix.getRows(), 0 );
      auto sssp_tnl = [&] () mutable {
         TNL::Algorithms::Graphs::singleSourceShortestPath( adjacencyMatrix, 0, ssspDistances );
      };
      benchmark.time< Device >( device, sssp_tnl );

      if( ssspDistances != this->boostSSSPDistances )
      {
         std::cout << "ERROR: Distances do not match!" << std::endl;
         this->errors++;
      }
   }

   void boostBenchmarks( const HostMatrix& adjacencyMatrix, TNL::Benchmarks::Benchmark<>& benchmark )
   {
      benchmark.setMetadataColumns(
      TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                       { "device", "sequential" },
                                                       { "algorithm", std::string( "BFS Boost" ) } } ) );
      BoostGraph< Index, Real > boostGraph( adjacencyMatrix );

      std::vector<Index> boostBfsDistances( adjacencyMatrix.getRows() );
      auto bfs_boost = [&] () mutable {
         boostGraph.breadthFirstSearch( 0, boostBfsDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost );

      HostIndexVector boost_bfs_dist( boostBfsDistances );
      boost_bfs_dist.forAllElements( [] __cuda_callable__ ( Index i, Index& x ) { x = x == std::numeric_limits< Index >::max() ? -1 : x; } );
      this->boostBfsDistances = boost_bfs_dist;


      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType<Real>() },
                                                          { "device", "sequential" },
                                                          { "algorithm", std::string( "SSSP Boost" ) } } ) );
      std::vector<Real> boostSSSPDistances( adjacencyMatrix.getRows() );
      auto sssp_boost = [&] () mutable {
         boostGraph.singleSourceShortestPath( 0, boostSSSPDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", sssp_boost );
      HostRealVector boost_sssp_dist( boostSSSPDistances );
      boost_sssp_dist.forAllElements( [] __cuda_callable__ ( Index i, Real& x ) { x = x == std::numeric_limits< Real >::max() ? -1 : x; } );
      this->boostSSSPDistances = boost_sssp_dist;
   }

#ifdef HAVE_GUNROCK
   void gunrockBenchmarks( const HostMatrix& adjacencyMatrix, TNL::Benchmarks::Benchmark<>& benchmark, std::string filename )
   {
      TNL_ASSERT_TRUE( adjacencyMatrix.getRows() == adjacencyMatrix.getColumns(), "Adjacency matrix must be square matrix." );

      std::vector<int> row_offsets;
      std::vector<int> column_indices, values;

      TNL::copy( adjacencyMatrix.getSegments().getOffsets(), row_offsets );
      TNL::copy( adjacencyMatrix.getColumnIndexes(), column_indices );
      TNL::copy( adjacencyMatrix.getValues(), values );

      thrust::device_vector< Index > d_row_offsets( adjacencyMatrix.getRows() + 1 );
      thrust::device_vector< Index > d_column_indices( adjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< Real > d_values( adjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< Index > d_row_indices( adjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< Index > d_column_offsets( adjacencyMatrix.getColumns() + 1 );
      thrust::copy( row_offsets.begin(), row_offsets.end(), d_row_offsets.begin() );
      thrust::copy( column_indices.begin(), column_indices.end(), d_column_indices.begin() );
      thrust::copy( values.begin(), values.end(), d_values.begin() );

      //std::cout << "d_column_indices = ";
      //thrust::copy(d_values.begin(), d_values.end(),
      //         std::ostream_iterator<int>(std::cout, " "));

      auto graph =
        gunrock::graph::build::from_csr<gunrock::memory_space_t::device, gunrock::graph::view_t::csr >(
            adjacencyMatrix.getRows(),                 // rows
            adjacencyMatrix.getColumns(),              // columns
            adjacencyMatrix.getNonzeroElementsCount(), // nonzeros
            d_row_offsets.data().get(),                // row_offsets
            d_column_indices.data().get(),             // column_indices
            d_values.data().get(),                     // values
            d_row_indices.data().get(),                // row_indices
            d_column_offsets.data().get()              // column_offsets
        );

      GunrockBenchmark< Real, Index > gunrockBenchmark;

      // Benchmarking breadth-first search
      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "algorithm", std::string( "BFS Gunrock" ) } } ) );
      std::vector< Index > bfsDistances( adjacencyMatrix.getRows() );
      Index start = 0;
      gunrockBenchmark.breadthFirstSearch( benchmark, graph, start, adjacencyMatrix.getRows(), bfsDistances );
      //TNL::Containers::Vector< Index > bfsDistances( distances );
      //std::cout << " distances: " << bfsDistances << std::endl;
      std::vector< Real > ssspDistances( adjacencyMatrix.getRows() );
      gunrockBenchmark.singleSourceShortestPath( benchmark, graph, start, adjacencyMatrix.getRows(), ssspDistances );
   }
#endif

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      auto inputFile = parameters.getParameter< TNL::String >( "input-file" );
      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const int loops = parameters.getParameter< int >("loops");
      const int verbose = parameters.getParameter< int >("verbose");

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, loops, verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      this->errors = 0;

      TNL::String device = parameters.getParameter< TNL::String >( "device" );

      std::cout << "Graphs benchmark  with " << TNL::getType<Real>() << " precision and device: " << device << std::endl;

      HostMatrix adjacencyMatrix;
      std::cout << "Reading graph from file " << inputFile << std::endl;
      TNL::Algorithms::Graphs::GraphReader< HostMatrix >::readEdgeList( inputFile, adjacencyMatrix );

      boostBenchmarks( adjacencyMatrix, benchmark );
      gunrockBenchmarks( adjacencyMatrix, benchmark, inputFile );

      if( device == "sequential" || device == "all" )
         TNLBenchmarks< TNL::Devices::Sequential >( adjacencyMatrix, benchmark, "sequential" );
      if( device == "host" || device == "all" )
         TNLBenchmarks< TNL::Devices::Host >( adjacencyMatrix, benchmark, "host" );
#ifdef __CUDACC__
      if( device == "cuda" || device == "all" )
         TNLBenchmarks< TNL::Devices::Cuda >( adjacencyMatrix, benchmark, "cuda" );
#endif

      return true;
   }

protected:

   // These vectors serve as a reference solution for comparison with TNL
   HostIndexVector boostBfsDistances;
   HostRealVector boostSSSPDistances;
   int errors;
};

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

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct GraphsBenchmark
{
   static void configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("input-file", "Input file with the graph." );
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-graphs.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");

      config.addEntry<int>("loops", "Number of iterations for every computation.", 10);
      config.addEntry<int>("verbose", "Verbose mode.", 1);
   }

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      using Matrix = TNL::Matrices::SparseMatrix<Real, Device, Index>;
      using IndexVector = TNL::Containers::Vector<Index, Device, Index>;
      using RealVector = TNL::Containers::Vector<Real, Device, Index>;
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


      auto precision = TNL::getType<Real>();
      TNL::String device;
      if( std::is_same< Device, TNL::Devices::Sequential >::value )
         device = "sequential";
      if( std::is_same< Device, TNL::Devices::Host >::value )
         device = "host";
      if( std::is_same< Device, TNL::Devices::Cuda >::value )
         device = "cuda";

      std::cout << "Graphs benchmark  with (" << precision << ", " << device << ")" << std::endl;

      Matrix adjacencyMatrix;
      std::cout << "Reading graph from file " << inputFile << std::endl;
      TNL::Algorithms::Graphs::GraphReader< Matrix >::readEdgeList( inputFile, adjacencyMatrix );
      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", precision },
                                                          { "device", device },
                                                          { "algorithm", std::string( "BFS TNL" ) } } ) );

      // Benchmarking breadth-first search
      IndexVector bfsDistances( adjacencyMatrix.getRows(), 0 );
      auto bfs_tnl = [&] () mutable {
         TNL::Algorithms::Graphs::breadthFirstSearch( adjacencyMatrix, 0, bfsDistances );
      };
      benchmark.time< Device >( device, bfs_tnl );

      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", precision },
                                                          { "device", device },
                                                          { "algorithm", std::string( "BFS Boost" ) } } ) );
      BoostGraph< Index, Real > boostGraph( adjacencyMatrix );
      std::vector< Index > boostBfsDistances;
      auto bfs_boost = [&] () mutable {
         boostGraph.breadthFirstSearch( 0, boostBfsDistances );
      };
      benchmark.time< Device >( device, bfs_boost );

      IndexVector boost_bfs_dist( boostBfsDistances );
      boost_bfs_dist.forAllElements( [] __cuda_callable__ ( Index i, Index& x ) { x = x == std::numeric_limits< Index >::max() ? -1 : x; } );
      if( bfsDistances != boost_bfs_dist )
      {
         std::cout << "ERROR: Distances do not match!" << std::endl;
         return false;
      }

      // Benchmarking single-source shortest paths
      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", precision },
                                                          { "device", device },
                                                          { "algorithm", std::string( "SSSP TNL" ) } } ) );

      RealVector ssspDistances( adjacencyMatrix.getRows(), 0 );
      auto sssp_tnl = [&] () mutable {
         TNL::Algorithms::Graphs::singleSourceShortestPath( adjacencyMatrix, 0, ssspDistances );
      };
      benchmark.time< Device >( device, sssp_tnl );

      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", precision },
                                                          { "device", device },
                                                          { "algorithm", std::string( "SSSP Boost" ) } } ) );
      std::vector< Real > boostSSSPDistances;
      auto sssp_boost = [&] () mutable {
         boostGraph.singleSourceShortestPath( 0, boostSSSPDistances );
      };
      benchmark.time< Device >( device, sssp_boost );
      RealVector boost_sssp_dist( boostSSSPDistances );
      boost_sssp_dist.forAllElements( [] __cuda_callable__ ( Index i, Real& x ) { x = x == std::numeric_limits< Real >::max() ? -1 : x; } );
      if( ssspDistances != boost_sssp_dist )
      {
         std::cout << "ERROR: Distances do not match!" << std::endl;
         return false;
      }
      return true;
   }

protected:

   Real xDomainSize = 0.0, yDomainSize = 0.0;
   Real alpha = 0.0, beta = 0.0, gamma = 0.0;
   Real timeStep = 0.0, finalTime = 0.0;
   bool outputData = false;
   bool verbose = false;
   Index maxIterations = 0;

   TNL::Containers::Vector<Real, Device> ux, aux;
};

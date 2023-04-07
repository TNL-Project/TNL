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
      using Vector = TNL::Containers::Vector<Index, Device, Index>;
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
      Vector distances( adjacencyMatrix.getRows(), 0 );

      TNL::Algorithms::Graphs::breadthFirstSearch( adjacencyMatrix, 0, distances );

      BoostGraph boostGraph( adjacencyMatrix );
      std::vector< Index > boostDistances;
      boostGraph.breadthFirstSearch( 0, boostDistances );
      Vector boost_v( boostDistances );
      if( distances != boost_v )
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

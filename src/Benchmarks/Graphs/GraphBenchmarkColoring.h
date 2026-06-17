// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"

#include <TNL/Graphs/Algorithms/graphColoring.h>

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
struct GraphBenchmarkColoring : public GraphBenchmarkBase< Real, Index, GraphBenchmarkColoring< Real, Index > >
{
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkColoring< Real, Index > >;
   using typename Base::HostDigraph;
   using typename Base::HostGraph;
   using typename Base::HostIndexVector;
   using typename Base::IndexType;
   using SequentialGraph = TNL::Graphs::Graph< Real, TNL::Devices::Sequential, Index, TNL::Graphs::UndirectedGraph >;
   using SequentialIndexVector = TNL::Containers::Vector< Index, TNL::Devices::Sequential, Index >;

   GraphBenchmarkColoring( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {}

   void
   runOtherBenchmarks( const HostDigraph&, const HostGraph& graph, IndexType, IndexType, TNL::Benchmarks::Benchmark& benchmark )
   {
      SequentialGraph sequentialGraph( graph );
      SequentialIndexVector sequentialColors( sequentialGraph.getVertexCount(), 0 );
      SequentialIndexVector sequentialLubyColors( sequentialGraph.getVertexCount(), 0 );

      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "solver", "TNL" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "N/A" } );

      benchmark.setMetadataElement( { "problem", "coloring seq" } );
      auto coloringSequential = [ & ]() mutable
      {
         TNL::Graphs::Algorithms::graphColoring( sequentialGraph, sequentialColors );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", coloringSequential );

      if( ! TNL::Graphs::Algorithms::isProperlyColored( sequentialGraph, sequentialColors ) ) {
         std::cout << "Sequential graph coloring produced an invalid coloring.\n";
         this->errors++;
      }

      benchmark.setMetadataElement( { "problem", "coloring luby seq" } );
      auto coloringLubySequential = [ & ]() mutable
      {
         TNL::Graphs::Algorithms::graphColoringLuby( sequentialGraph, sequentialLubyColors );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", coloringLubySequential );

      if( ! TNL::Graphs::Algorithms::isProperlyColored( sequentialGraph, sequentialLubyColors ) ) {
         std::cout << "Sequential Luby graph coloring produced an invalid coloring.\n";
         this->errors++;
      }
   }

   template< typename Digraph, typename Graph >
   void
   runTNLAlgorithm(
      Digraph&,
      Graph& graph,
      IndexType,
      IndexType,
      TNL::Benchmarks::Benchmark& benchmark,
      const TNL::String& device,
      const TNL::String& segments )
   {
      using Device = typename std::remove_reference_t< decltype( graph ) >::DeviceType;
      using SequentialDevice = TNL::Devices::Sequential;
      using Vector = TNL::Containers::Vector< Index, Device, Index >;

      if constexpr( std::is_same_v< Device, SequentialDevice > ) {
         return;
      }
      else {
         Vector colors( graph.getVertexCount(), 0 );
         Vector lubiColors( graph.getVertexCount(), 0 );
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "kernel", segments } );

         benchmark.setMetadataElement( { "problem", "coloring" } );
         benchmark.setMetadataElement( { "launch cfg.", "N/A" } );
         auto coloring = [ & ]() mutable
         {
            TNL::Graphs::Algorithms::graphColoring( graph, colors );
         };
         benchmark.time< Device >( device, coloring );

         if( ! TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) ) {
            std::cout << "Graph coloring produced an invalid coloring on device " << device << ".\n";
            this->errors++;
         }

         benchmark.setMetadataElement( { "problem", "coloring luby" } );
         benchmark.setMetadataElement( { "launch cfg.", "N/A" } );
         auto coloringLuby = [ & ]() mutable
         {
            TNL::Graphs::Algorithms::graphColoringLuby( graph, lubiColors );
         };
         benchmark.time< Device >( device, coloringLuby );

         if( ! TNL::Graphs::Algorithms::isProperlyColored( graph, lubiColors ) ) {
            std::cout << "Luby graph coloring produced an invalid coloring on device " << device << ".\n";
            this->errors++;
         }
      }
   }
};

}  // namespace TNL::Benchmarks::Graphs

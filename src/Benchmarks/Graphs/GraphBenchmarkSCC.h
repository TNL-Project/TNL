// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"

#include <TNL/Graphs/Algorithms/stronglyConnectedComponents.h>

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
struct GraphBenchmarkSCC : public GraphBenchmarkBase< Real, Index, GraphBenchmarkSCC< Real, Index > >
{
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkSCC< Real, Index > >;
   using typename Base::HostDigraph;
   using typename Base::HostGraph;
   using typename Base::HostIndexVector;
   using typename Base::IndexType;
   using SequentialDigraph = TNL::Graphs::Graph< Real, TNL::Devices::Sequential, Index, TNL::Graphs::DirectedGraph >;
   using SequentialIndexVector = TNL::Containers::Vector< Index, TNL::Devices::Sequential, Index >;

   GraphBenchmarkSCC( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {}

   void
   runOtherBenchmarks(
      const HostDigraph& digraph,
      const HostGraph&,
      IndexType,
      IndexType,
      TNL::Benchmarks::Benchmark& benchmark )
   {
      SequentialDigraph sequentialDigraph( digraph );
      SequentialIndexVector sequentialReferenceComponents( sequentialDigraph.getVertexCount(), -1 );

      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "solver", "TNL" } );
      benchmark.setMetadataElement( { "problem", "SCC seq" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "N/A" } );

      auto sccSequential = [ & ]() mutable
      {
         TNL::Graphs::Algorithms::stronglyConnectedComponents( sequentialDigraph, sequentialReferenceComponents );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", sccSequential );

      referenceComponents = sequentialReferenceComponents;
   }

   template< typename Digraph, typename Graph >
   void
   runTNLAlgorithm(
      Digraph& digraph,
      Graph&,
      IndexType,
      IndexType,
      TNL::Benchmarks::Benchmark& benchmark,
      const TNL::String& device,
      const TNL::String& segments )
   {
      using Device = typename std::remove_reference_t< decltype( digraph ) >::DeviceType;
      using SequentialDevice = TNL::Devices::Sequential;
      using Vector = TNL::Containers::Vector< Index, Device, Index >;

      if constexpr( std::is_same_v< Device, SequentialDevice > ) {
         return;
      }
      else {
         Vector deviceReferenceComponents( digraph.getVertexCount(), -1 );
         deviceReferenceComponents = referenceComponents;

         Vector components( digraph.getVertexCount(), -1 );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "SCC dir" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& launchEntry :
              Algorithms::Segments::traversingLaunchConfigurations( digraph.getAdjacencyMatrix().getSegments() ) )
         {
            const auto& launchConfig = launchEntry.first;
            const auto& tag = launchEntry.second;

            benchmark.setMetadataElement( { "launch cfg.", tag } );

            auto scc = [ &, launchConfig ]() mutable
            {
               TNL::Graphs::Algorithms::stronglyConnectedComponents( digraph, components, launchConfig );
            };
            benchmark.time< Device >( device, scc );

            if( components != deviceReferenceComponents ) {
               std::cout << "Strongly connected components mismatch for production implementation.\n";
               this->errors++;
            }
         }
      }
   }

   HostIndexVector referenceComponents;
};

}  // namespace TNL::Benchmarks::Graphs

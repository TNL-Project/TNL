// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"
#include <TNL/Graphs/Algorithms/connectedComponents.h>

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
struct GraphBenchmarkCC : public GraphBenchmarkBase< Real, Index, GraphBenchmarkCC< Real, Index > >
{
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkCC< Real, Index > >;
   using typename Base::HostDigraph;
   using typename Base::HostGraph;
   using typename Base::HostIndexVector;
   using typename Base::IndexType;

   GraphBenchmarkCC( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {}

   void
   runOtherBenchmarks( const HostDigraph&, const HostGraph& graph, IndexType, IndexType, TNL::Benchmarks::Benchmark& benchmark )
   {
      referenceComponents.setSize( graph.getVertexCount() );
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "solver", "TNL" } );
      benchmark.setMetadataElement( { "problem", "CC seq" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "N/A" } );

      auto ccSequential = [ & ]() mutable
      {
         TNL::Graphs::Algorithms::connectedComponents( graph, referenceComponents );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", ccSequential );
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
      using Vector = TNL::Containers::Vector< Index, Device, Index >;
      using SequentialDevice = TNL::Devices::Sequential;

      if constexpr( std::is_same_v< Device, SequentialDevice > ) {
         return;
      }
      else {
         Vector deviceReferenceComponents( graph.getVertexCount(), -1 );
         deviceReferenceComponents = referenceComponents;

         Vector components( graph.getVertexCount(), -1 );
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "CC par" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& launchEntry :
              Algorithms::Segments::traversingLaunchConfigurations( graph.getAdjacencyMatrix().getSegments() ) )
         {
            const auto& launchConfig = launchEntry.first;
            const auto& tag = launchEntry.second;

            benchmark.setMetadataElement( { "launch cfg.", tag } );

            auto cc = [ &, launchConfig ]() mutable
            {
               TNL::Graphs::Algorithms::connectedComponents( graph, components, launchConfig );
            };
            benchmark.time< Device >( device, cc );

            if( components != deviceReferenceComponents ) {
               std::cout << "Connected components mismatch for production implementation.\n";
               this->errors++;
            }
         }
      }
   }

   HostIndexVector referenceComponents;
};

}  // namespace TNL::Benchmarks::Graphs

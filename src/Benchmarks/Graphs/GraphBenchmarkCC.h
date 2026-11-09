// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "GraphBenchmarkBase.h"
#ifdef HAVE_GAPBS
   #include "GapbsKernels.h"
#endif
#include <TNL/Graphs/Algorithms/connectedComponents.h>

namespace TNL::Benchmarks::Graphs {

/**
 * \brief Connected components benchmark following the GAP Benchmark Suite.
 *
 * The components are computed on the undirected graph (for directed graphs, GAP requires the weakly connected
 * components). The kernel is run in 16 trials, each including the allocation of the output, and the output is
 * verified against the number of components.
 */
template< typename Real = double, typename Index = int >
class GraphBenchmarkCC : public GraphBenchmarkBase< Real, Index, GraphBenchmarkCC< Real, Index > >
{
public:
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkCC< Real, Index > >;
   using typename Base::HostDigraph;
   using typename Base::HostGraph;
   using typename Base::HostIndexVector;
   using typename Base::IndexType;

   static constexpr int defaultTrials = 16;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      Base::configSetup( config );
      config.addDelimiter( "CC benchmark settings:" );
      config.addEntry< bool >( "with-gapbs", "Run the GAP reference code (gapbs) benchmarks.", true );
   }

   GraphBenchmarkCC( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {
      withGapbs = this->parameters.template getParameter< bool >( "with-gapbs" );
   }

   void
   runOtherBenchmarks( const HostDigraph&, const HostGraph& graph, IndexType, IndexType, TNL::Benchmarks::Benchmark& benchmark )
   {
      if( ! this->runUndirected ) {
         std::cout << "The connected components are computed on the undirected graph, use --orientation undirected or "
                      "both.\n";
         return;
      }
      std::cout << "Counting the connected components for the verification ...\n";
      components = countConnectedComponents( graph.getAdjacencyMatrix() );
      std::cout << "The graph has " << components << " connected components.\n";

      if( withGapbs )
         runGapbsCC( graph, benchmark );

      HostIndexVector sequentialComponents;
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "solver", "TNL" } );
      benchmark.setMetadataElement( { "problem", "CC seq" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "N/A" } );
      auto reset = [ & ]()
      {
         sequentialComponents.reset();
      };
      auto ccSequential = [ & ]()
      {
         TNL::Graphs::Algorithms::connectedComponents( graph, sequentialComponents );
      };
      auto check = [ & ]()
      {
         return verify( graph, sequentialComponents.getSize(), sequentialComponents.getData() );
      };
      this->template timeWholeGraphTrials< TNL::Devices::Sequential >(
         benchmark, "TNL", "TNL CC seq", reset, ccSequential, check );
   }

   template< typename Label >
   GraphVerificationResult
   verify( const HostGraph& graph, std::size_t size, const Label* labels ) const
   {
      if( size != static_cast< std::size_t >( graph.getVertexCount() ) ) {
         GraphVerificationResult result;
         result.violations = 1;
         result.message = "the output has a wrong size";
         return result;
      }
      return verifyCC( graph.getAdjacencyMatrix(), labels, components );
   }

   void
   runGapbsCC( const HostGraph& graph, TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_GAPBS
      if( ! this->gapInput || this->parameters.template getParameter< TNL::String >( "input-file" ).endsWith( ".wsg" ) ) {
         std::cout << "Skipping gapbs, it needs an .sg input file of the GAP benchmark.\n";
         return;
      }
      try {
         Gapbs::CC gapbs( this->undirectedInputFile );
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "solver", "gapbs" } );
         benchmark.setMetadataElement( { "problem", "CC par" } );
         benchmark.setMetadataElement( { "kernel", "N/A" } );
         benchmark.setMetadataElement( { "launch cfg.", "Afforest" } );
         std::vector< std::int32_t > labels;
         auto reset = [ & ]()
         {
            gapbs.release();
         };
         auto compute = [ & ]()
         {
            gapbs.run();
         };
         auto check = [ & ]()
         {
            gapbs.getComponents( labels );
            return verify( graph, labels.size(), labels.data() );
         };
         this->template timeWholeGraphTrials< TNL::Devices::Host >( benchmark, "gapbs", "gapbs CC", reset, compute, check );
      }
      catch( const std::exception& e ) {
         std::cerr << "gapbs CC failed: " << e.what() << '\n';
         this->errors++;
      }
#endif
   }

   template< typename Digraph, typename Graph >
   void
   runTNLAlgorithm(
      Digraph&,
      Graph& graph,
      IndexType,
      IndexType,
      TNL::Benchmarks::Benchmark& benchmark,
      const TNL::String& segments )
   {
      using Device = typename std::remove_reference_t< decltype( graph ) >::DeviceType;
      using Vector = TNL::Containers::Vector< Index, Device, Index >;
      using SequentialDevice = TNL::Devices::Sequential;

      if constexpr( std::is_same_v< Device, SequentialDevice > ) {
         return;
      }
      else {
         if( ! this->runUndirected )
            return;
         Vector labels;
         HostIndexVector hostLabels;
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "CC par" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& launchEntry :
              Algorithms::Segments::traversingLaunchConfigurations( graph.getAdjacencyMatrix().getSegments() ) )
         {
            const auto& launchConfig = launchEntry.first;
            const auto& tag = launchEntry.second;

            benchmark.setMetadataElement( { "launch cfg.", tag } );

            auto reset = [ & ]()
            {
               // the output is allocated by connectedComponents, so the allocation is measured
               labels.reset();
            };
            auto cc = [ &, launchConfig ]()
            {
               TNL::Graphs::Algorithms::connectedComponents( graph, labels, launchConfig );
            };
            auto check = [ & ]()
            {
               hostLabels = labels;
               return verify( *this->hostGraphPointer, hostLabels.getSize(), hostLabels.getData() );
            };
            this->template timeWholeGraphTrials< Device >(
               benchmark, "TNL", "TNL CC par " + std::string( segments ) + " " + tag, reset, cc, check );
         }
      }
   }

protected:
   bool withGapbs;
   //! \brief Number of the connected components, computed for the verification.
   std::int64_t components = 0;
};

}  // namespace TNL::Benchmarks::Graphs

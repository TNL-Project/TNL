// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/TypeInfo.h>

#include "GNNLayers.h"

namespace TNL::Benchmarks::GNN {

/**
 * \brief Benchmark result extended with the checksums of the output, which are compared with the reference
 * implementations in `gnn-reference.py`.
 */
struct GNNResult : public BenchmarkResult
{
   explicit GNNResult( std::function< std::pair< double, double >() > checksums )
   : checksums( std::move( checksums ) )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.push_back( "output sum" );
      headers.push_back( "output norm" );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      const auto [ sum, norm ] = checksums();
      elements << sum << norm;
      return elements;
   }

   std::function< std::pair< double, double >() > checksums;
};

template< typename Layer, typename Device, typename Real, typename Index >
void
benchmarkModel( Benchmark& benchmark,
                const Config::ParameterContainer& parameters,
                const Matrices::SparseMatrix< Real, Devices::Host, Index >& hostA,
                const std::string& modelName )
{
   const Index inputs = parameters.getParameter< int >( "input-features" );
   const Index hidden = parameters.getParameter< int >( "hidden-features" );
   const Index outputs = parameters.getParameter< int >( "output-features" );

   Matrices::SparseMatrix< Real, Device, Index > A;
   A = hostA;
   FeaturesType< Real, Device, Index > X( A.getRows(), inputs );
   fillRandom( X, 1, 1.0 );
   FeaturesType< Real, Device, Index > output;
   TwoLayerModel< Layer > model( inputs, hidden, outputs );

   GNNResult result(
      [ & ]()
      {
         return std::pair< double, double >( TNL::sum( output.getValues() ),
                                             std::sqrt( TNL::dot( output.getValues(), output.getValues() ) ) );
      } );
   benchmark.setMetadataElement( { "model", modelName } );
   const std::string vendor = std::is_same_v< Device, Devices::Host > ? "Eigen" : "cuBLAS";
   for( const auto& [ gemm, performer ] : { std::pair{ Gemm::TNL, std::string( "TNL" ) },
                                           std::pair{ Gemm::Vendor, "TNL + " + vendor } } )
   {
      auto compute = [ &, gemm = gemm ]()
      {
         model.forward( A, X, output, gemm );
      };
      try {
         benchmark.setMetadataElement( { "part", "forward" } );
         benchmark.time< Device >( performer, compute, result );
      }
      catch( const std::exception& e ) {
         benchmark.addErrorMessage( performer + ": " + e.what() );
      }
   }
   // only the aggregations, i.e. the sparse-dense products with the fused bias, root term and activation
   auto aggregate = [ & ]()
   {
      model.aggregate( A, output );
   };
   benchmark.setMetadataElement( { "part", "aggregation" } );
   benchmark.time< Device >( "TNL", aggregate, result );
}

template< typename Device, typename Real, typename Index >
void
benchmarkDevice( Benchmark& benchmark,
                 const Config::ParameterContainer& parameters,
                 const Matrices::SparseMatrix< Real, Devices::Host, Index >& Ahat,
                 const Matrices::SparseMatrix< Real, Devices::Host, Index >& Amean )
{
   const auto models = parameters.getList< std::string >( "models" );
   auto isSelected = [ & ]( const std::string& model )
   {
      return std::find( models.begin(), models.end(), "all" ) != models.end()
          || std::find( models.begin(), models.end(), model ) != models.end();
   };
   if( isSelected( "gcn" ) )
      benchmarkModel< GCNLayer< Real, Device, Index >, Device >( benchmark, parameters, Ahat, "gcn" );
   if( isSelected( "sage" ) )
      benchmarkModel< SAGELayer< Real, Device, Index >, Device >( benchmark, parameters, Amean, "sage" );
}

template< typename Real, typename Index >
void
runBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   using HostMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index >;
   const auto& inputFile = parameters.getParameter< std::string >( "input-file" );
   const auto& device = parameters.getParameter< std::string >( "device" );
   HostMatrix A;
   Matrices::MatrixReader< HostMatrix >::readMtx( inputFile, A );
   if( A.getRows() != A.getColumns() )
      throw std::invalid_argument( "the adjacency matrix of the graph must be square" );
   const HostMatrix Ahat = normalizeGCN( A );
   const HostMatrix Amean = normalizeSAGE( A );

   benchmark.setMetadataColumns( Benchmark::MetadataColumns( {
      { "graph", inputFile },
      { "nodes", std::to_string( A.getRows() ) },
      { "edges", std::to_string( A.getNonzeroElementsCount() ) },
      { "precision", getType< Real >() },
      { "input features", std::to_string( parameters.getParameter< int >( "input-features" ) ) },
      { "hidden features", std::to_string( parameters.getParameter< int >( "hidden-features" ) ) },
      { "output features", std::to_string( parameters.getParameter< int >( "output-features" ) ) },
   } ) );
   benchmark.setOperation( "GNN inference" );

   if( device == "host" || device == "all" )
      benchmarkDevice< Devices::Host >( benchmark, parameters, Ahat, Amean );
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( device == "cuda" || device == "hip" || device == "all" )
      benchmarkDevice< Devices::GPU >( benchmark, parameters, Ahat, Amean );
#endif
}

}  // namespace TNL::Benchmarks::GNN

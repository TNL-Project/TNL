// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/TypeInfo.h>

#include "LinearSVMGradient.h"

namespace TNL::Benchmarks::SVM {

/**
 * \brief Benchmark result extended with the norm of the computed gradient.
 */
struct SVMResult : public BenchmarkResult
{
   explicit SVMResult( std::function< double() > norm )
   : norm( std::move( norm ) )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.push_back( "gradient norm" );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << norm();
      return elements;
   }

   std::function< double() > norm;
};

//! \brief Reads the classes of the samples, one per line.
template< typename Index >
std::vector< Index >
readLabels( const std::string& fileName )
{
   std::ifstream file( fileName );
   if( ! file )
      throw std::runtime_error( "cannot open the file " + fileName );
   std::vector< Index > labels;
   Index label;
   while( file >> label )
      labels.push_back( label );
   return labels;
}

/**
 * \brief Returns the initial weights `0.2 (h - 0.5)`, where `h` from `[0, 1)` is given by the hash splitmix64 of the
 * index of the element.
 *
 * The same initialization is used by the reference implementations in `svm-reference.py`.
 */
template< typename Real, typename Index >
Matrices::DenseMatrix< Real, Devices::Host, Index >
getInitialWeights( Index features, Index classes )
{
   Matrices::DenseMatrix< Real, Devices::Host, Index > W( features, classes );
   for( Index feature = 0; feature < features; feature++ )
      for( Index c = 0; c < classes; c++ ) {
         std::uint64_t z = ( std::uint64_t( feature ) * classes + c ) * 0x9E3779B97F4A7C15ULL;
         z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
         z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBULL;
         z ^= z >> 31;
         const double h = double( z >> 40 ) / double( 1 << 24 );
         W.setElement( feature, c, Real( 0.2 * ( h - 0.5 ) ) );
      }
   return W;
}

template< typename Device, typename Real, typename Index >
void
benchmarkDevice(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& hostX,
   const std::vector< Index >& hostLabels,
   Index classes )
{
   const auto methods = parameters.getList< std::string >( "method" );
   auto isSelected = [ & ]( const std::string& method )
   {
      return std::find( methods.begin(), methods.end(), "all" ) != methods.end()
          || std::find( methods.begin(), methods.end(), method ) != methods.end();
   };

   using Gradient = LinearSVMGradient< Real, Device, Index >;
   typename Gradient::MatrixType X;
   X = hostX;
   typename Gradient::IndexVectorType labels( hostLabels.size() );
   Containers::Vector< Index, Devices::Host, Index > modLabels( hostLabels.size() );
   for( std::size_t i = 0; i < hostLabels.size(); i++ )
      modLabels[ i ] = hostLabels[ i ] % classes;
   labels = modLabels;
   Gradient gradient( X, labels, classes, parameters.getParameter< double >( "cost" ) );
   typename Gradient::DenseMatrixType W;
   W = getInitialWeights< Real >( hostX.getColumns(), classes );
   gradient.setWeights( W );

   for( const auto& [ method, methodName ] :
        { std::pair{ Method::SpMV, "spmv" }, std::pair{ Method::SpMM, "spmm" }, std::pair{ Method::Fused, "spmm-fused" } } )
   {
      if( ! isSelected( methodName ) )
         continue;
      // structured bindings cannot be captured in C++17
      const Method selected = method;
      auto compute = [ & ]()
      {
         gradient.compute( selected );
      };
      SVMResult result(
         [ & ]()
         {
            return gradient.getGradientNorm( selected );
         } );
      benchmark.setMetadataElement( { "method", methodName } );
      benchmark.time< Device >( "TNL", compute, result );
   }
}

template< typename Real, typename Index >
void
runBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   using HostMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index >;
   const auto& inputFile = parameters.getParameter< std::string >( "input-file" );
   const auto& device = parameters.getParameter< std::string >( "device" );
   std::string labelsFile = parameters.getParameter< std::string >( "labels-file" );
   if( labelsFile.empty() )
      labelsFile = inputFile.substr( 0, inputFile.rfind( '.' ) ) + ".labels";
   HostMatrix X;
   Matrices::MatrixReader< HostMatrix >::readMtx( inputFile, X );
   const auto labels = readLabels< Index >( labelsFile );
   if( Index( labels.size() ) != X.getRows() )
      throw std::runtime_error( "the number of labels differs from the number of samples" );
   const Index datasetClasses = *std::max_element( labels.begin(), labels.end() ) + 1;
   const Index nonzeros = X.getNonzeroElementsCount();

   for( const int requestedClasses : parameters.getList< int >( "classes" ) ) {
      const Index classes = requestedClasses > 0 ? Index( requestedClasses ) : datasetClasses;
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "matrix", inputFile },
               { "samples", std::to_string( X.getRows() ) },
               { "features", std::to_string( X.getColumns() ) },
               { "nonzeros", std::to_string( nonzeros ) },
               { "precision", getType< Real >() },
               { "classes", std::to_string( classes ) },
            } ) );
      benchmark.setOperation( "SVM gradient" );
      // X and X^T are read once (the dense matrices are not counted)
      benchmark.setDatasetSize( 2 * std::size_t( nonzeros ) * ( sizeof( Real ) + sizeof( Index ) ) );
      benchmark.setOperationsPerLoop( 4 * std::size_t( nonzeros ) * classes );

      if( device == "host" || device == "all" )
         benchmarkDevice< Devices::Host >( benchmark, parameters, X, labels, classes );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" )
         benchmarkDevice< Devices::GPU >( benchmark, parameters, X, labels, classes );
#endif
   }
}

}  // namespace TNL::Benchmarks::SVM

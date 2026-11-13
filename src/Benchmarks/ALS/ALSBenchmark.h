// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Timer.h>
#include <TNL/TypeInfo.h>

#include "SparseALS.h"

namespace TNL::Benchmarks::ALS {

/**
 * \brief Benchmark result extended with the time of one iteration and the loss of the final factors.
 */
struct ALSResult : public BenchmarkResult
{
   ALSResult( int iterations, std::function< double() > loss )
   : iterations( iterations ),
     loss( std::move( loss ) )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.push_back( "time per iteration" );
      headers.push_back( "loss" );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << time / iterations << loss();
      return elements;
   }

   int iterations;
   std::function< double() > loss;
};

/**
 * \brief Returns the initial factors `(h - 0.5) / k` of `rows` users or items, where `h` from `[0, 1)` is given by the
 * hash splitmix64 of the index of the element and `seed` (1 for the users, 2 for the items).
 *
 * The same initialization is used by the reference implementations in `als-reference.py`.
 */
template< typename Real, typename Index >
Matrices::DenseMatrix< Real, Devices::Host, Index >
getInitialFactors( Index rows, Index factors, std::uint64_t seed )
{
   Matrices::DenseMatrix< Real, Devices::Host, Index > F( rows, factors );
   for( Index row = 0; row < rows; row++ )
      for( Index j = 0; j < factors; j++ ) {
         std::uint64_t z = ( std::uint64_t( row ) * factors + j ) * 0x9E3779B97F4A7C15ULL + seed;
         z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
         z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBULL;
         z ^= z >> 31;
         const double h = double( z >> 40 ) / double( 1 << 24 );
         F.setElement( row, j, Real( ( h - 0.5 ) / factors ) );
      }
   return F;
}

template< typename Device, typename Real, typename Index >
void
benchmarkDevice(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& hostR,
   Index factors )
{
   const int iterations = parameters.getParameter< int >( "iterations" );
   const Real regularization = parameters.getParameter< double >( "regularization" );
   const int cgSteps = parameters.getParameter< int >( "cg-steps" );

   using ALS = SparseALS< Real, Device, Index >;
   typename ALS::MatrixType R;
   R = hostR;
   typename ALS::DenseMatrixType X0;
   typename ALS::DenseMatrixType Y0;
   X0 = getInitialFactors< Real >( hostR.getRows(), factors, 1 );
   Y0 = getInitialFactors< Real >( hostR.getColumns(), factors, 2 );
   ALS als( R, factors, regularization, cgSteps );

   // the configurations of the products are selected by measuring them before the benchmark
   als.setFactors( X0, Y0 );
   Timer tuningTimer;
   tuningTimer.start();
   als.tuneProducts();
   tuningTimer.stop();

   auto compute = [ & ]()
   {
      als.setFactors( X0, Y0 );
      for( int i = 0; i < iterations; i++ )
         als.iterate();
   };
   ALSResult result(
      iterations,
      [ & ]()
      {
         return als.getLoss();
      } );
   const auto solvers = parameters.getList< std::string >( "solver" );
   auto isSelected = [ & ]( const std::string& solver )
   {
      return std::find( solvers.begin(), solvers.end(), "all" ) != solvers.end()
          || std::find( solvers.begin(), solvers.end(), solver ) != solvers.end();
   };
   if( isSelected( "spmm" ) ) {
      als.setFused( false );
      benchmark.setMetadataElement( { "solver", "spmm" } );
      benchmark.setMetadataElement( { "users configuration", als.getSide( false ).configuration.toString() } );
      benchmark.setMetadataElement( { "items configuration", als.getSide( true ).configuration.toString() } );
      benchmark.setMetadataElement( { "tuning time", std::to_string( tuningTimer.getRealTime() ) } );
      benchmark.time< Device >( "TNL", compute, result );
   }
   if( isSelected( "fused" ) && als.isFusedSupported() ) {
      als.setFused( true );
      benchmark.setMetadataElement( { "solver", "fused" } );
      benchmark.setMetadataElement( { "users configuration", "N/A" } );
      benchmark.setMetadataElement( { "items configuration", "N/A" } );
      benchmark.setMetadataElement( { "tuning time", "0" } );
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
   const int iterations = parameters.getParameter< int >( "iterations" );
   const int cgSteps = parameters.getParameter< int >( "cg-steps" );
   HostMatrix R;
   Matrices::MatrixReader< HostMatrix >::readMtx( inputFile, R );
   const Index nonzeros = R.getNonzeroElementsCount();

   for( const int factors : parameters.getList< int >( "factors" ) ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "matrix", inputFile },
               { "users", std::to_string( R.getRows() ) },
               { "items", std::to_string( R.getColumns() ) },
               { "nonzeros", std::to_string( nonzeros ) },
               { "precision", getType< Real >() },
               { "factors", std::to_string( factors ) },
               { "iterations", std::to_string( iterations ) },
               { "regularization", std::to_string( parameters.getParameter< double >( "regularization" ) ) },
               { "cg steps", std::to_string( cgSteps ) },
            } ) );
      benchmark.setOperation( "ALS" );
      // both sides compute cgSteps + 1 sampled products and sparse-dense products, each reads R once
      const std::size_t products = std::size_t( iterations ) * 2 * ( cgSteps + 1 );
      benchmark.setDatasetSize( products * 2 * nonzeros * ( sizeof( Real ) + sizeof( Index ) ) );
      // the multiplications and additions of the sampled and sparse-dense products
      benchmark.setOperationsPerLoop( products * 4 * std::size_t( nonzeros ) * factors );

      if( device == "host" || device == "all" )
         benchmarkDevice< Devices::Host >( benchmark, parameters, R, Index( factors ) );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" )
         benchmarkDevice< Devices::GPU >( benchmark, parameters, R, Index( factors ) );
#endif
   }
}

}  // namespace TNL::Benchmarks::ALS

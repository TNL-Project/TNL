// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/TypeInfo.h>

namespace TNL::Benchmarks::DenseSparse {

/**
 * \brief Benchmark result extended with the sum of the elements of the product, which is compared between the methods.
 */
struct DenseSparseResult : public BenchmarkResult
{
   explicit DenseSparseResult( std::function< double() > checksum )
   : checksum( std::move( checksum ) )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.push_back( "checksum" );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << checksum();
      return elements;
   }

   std::function< double() > checksum;
};

/**
 * \brief Returns the dense matrix `B` (`k x n`) with the elements `h - 0.5`, where `h` from `[0, 1)` is given by the hash
 * splitmix64 of the index `i * n + j` of the element.
 *
 * The same matrix is used by the reference implementations in `dense-sparse-reference.py`.
 */
template< typename Real, typename Index, Algorithms::Segments::ElementsOrganization Organization >
Matrices::DenseMatrix< Real, Devices::Host, Index, Organization >
getDenseMatrix( Index k, Index n )
{
   Matrices::DenseMatrix< Real, Devices::Host, Index, Organization > B( k, n );
   for( Index i = 0; i < k; i++ )
      for( Index j = 0; j < n; j++ ) {
         std::uint64_t z = ( std::uint64_t( i ) * n + j ) * 0x9E3779B97F4A7C15ULL;
         z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
         z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBULL;
         z ^= z >> 31;
         B.setElement( i, j, Real( double( z >> 40 ) / double( 1 << 24 ) - 0.5 ) );
      }
   return B;
}

template< typename Device, Algorithms::Segments::ElementsOrganization Organization, typename Real, typename Index >
void
benchmarkDevice(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& hostA,
   Index k )
{
   const auto methods = parameters.getList< std::string >( "method" );
   auto isSelected = [ & ]( const std::string& method )
   {
      return std::find( methods.begin(), methods.end(), "all" ) != methods.end()
          || std::find( methods.begin(), methods.end(), method ) != methods.end();
   };
   using SparseMatrixType = Matrices::SparseMatrix< Real, Device, Index >;
   using DenseMatrixType = Matrices::DenseMatrix< Real, Device, Index, Organization >;

   SparseMatrixType A;
   A = hostA;
   // A^T is precomputed for the methods which use it
   SparseMatrixType AT;
   AT.getTransposition( A );
   DenseMatrixType B;
   B = getDenseMatrix< Real, Index, Organization >( k, hostA.getRows() );
   DenseMatrixType C( k, hostA.getColumns() );
   // the auxiliary matrices of the methods with the transpositions
   DenseMatrixType BT;
   DenseMatrixType CT( hostA.getColumns(), k );
   SparseMatrixType transposedA;

   const std::vector< std::pair< std::string, std::function< void() > > > computations{
      // the transpositions of both matrices, C^T = A^T * B^T, and the transposition of C^T
      { "transpose-all",
        [ & ]()
        {
           BT.getTransposition( B );
           transposedA.getTransposition( A );
           Matrices::sparseDenseMatrixProduct( transposedA, BT, CT );
           C.getTransposition( CT );
        } },
      // the same with precomputed A^T, e.g. when the product with the same sparse matrix is computed repeatedly
      { "transpose-dense",
        [ & ]()
        {
           BT.getTransposition( B );
           Matrices::sparseDenseMatrixProduct( AT, BT, CT );
           C.getTransposition( CT );
        } },
      // denseSparseMatrixProduct with precomputed A^T, the reduction in the rows of A^T without the dense transpositions
      { "transposed-A",
        [ & ]()
        {
           Matrices::denseSparseMatrixProduct( B, AT, C, Real( 1 ), Real( 0 ), Matrices::TransposeState::Transpose );
        } },
      // denseSparseMatrixProduct with A, the atomic additions to the columns of C
      { "atomic",
        [ & ]()
        {
           Matrices::denseSparseMatrixProduct( B, A, C );
        } },
   };
   for( const auto& [ method, computation ] : computations ) {
      if( ! isSelected( method ) )
         continue;
      DenseSparseResult result(
         [ & ]()
         {
            // the sum in double precision does not depend on the order of the elements
            Containers::Vector< Real, Devices::Host, Index > values;
            values = C.getValues();
            double sum = 0;
            for( Index i = 0; i < values.getSize(); i++ )
               sum += values[ i ];
            return sum;
         } );
      benchmark.setMetadataElement( { "method", method } );
      benchmark.time< Device >( "TNL", computation, result );
   }
}

template< typename Device, typename Real, typename Index >
void
benchmarkOrganizations(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& A,
   Index k )
{
   const auto& organization = parameters.getParameter< std::string >( "organization" );
   if( organization == "row-major" || organization == "all" ) {
      benchmark.setMetadataElement( { "organization", "row-major" } );
      benchmarkDevice< Device, Algorithms::Segments::RowMajorOrder >( benchmark, parameters, A, k );
   }
   if( organization == "column-major" || organization == "all" ) {
      benchmark.setMetadataElement( { "organization", "column-major" } );
      benchmarkDevice< Device, Algorithms::Segments::ColumnMajorOrder >( benchmark, parameters, A, k );
   }
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
   const Index nonzeros = A.getNonzeroElementsCount();

   for( const int k : parameters.getList< int >( "dense-rows" ) ) {
      benchmark.setMetadataColumns(
         Benchmark::MetadataColumns(
            {
               { "matrix", inputFile },
               { "rows", std::to_string( A.getRows() ) },
               { "columns", std::to_string( A.getColumns() ) },
               { "nonzeros", std::to_string( nonzeros ) },
               { "precision", getType< Real >() },
               { "dense rows", std::to_string( k ) },
            } ) );
      benchmark.setOperation( "dense-sparse product" );
      // A is read once, B is read and C is written once
      benchmark.setDatasetSize(
         std::size_t( nonzeros ) * ( sizeof( Real ) + sizeof( Index ) )
         + std::size_t( k ) * ( A.getRows() + A.getColumns() ) * sizeof( Real ) );
      benchmark.setOperationsPerLoop( 2 * std::size_t( nonzeros ) * k );

      if( device == "host" || device == "all" )
         benchmarkOrganizations< Devices::Host >( benchmark, parameters, A, Index( k ) );
#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" )
         benchmarkOrganizations< Devices::GPU >( benchmark, parameters, A, Index( k ) );
#endif
   }
}

}  // namespace TNL::Benchmarks::DenseSparse

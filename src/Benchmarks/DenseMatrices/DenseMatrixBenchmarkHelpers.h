// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrix.h>

#include <cmath>
#include <string>

namespace TNL::Benchmarks::DenseMatrices {

using TransposeState = TNL::Matrices::TransposeState;

template< typename Real, typename Index >
void
setMetadata(
   TNL::Benchmarks::Benchmark& benchmark,
   const std::string& algorithm,
   const std::string& matrix1Size,
   const std::string& matrix2Size )
{
   benchmark.setMetadataColumns(
      TNL::Benchmarks::Benchmark::MetadataColumns(
         { { "index type", TNL::getType< Index >() },
           { "real type", TNL::getType< Real >() },
           { "algorithm", algorithm },
           { "matrix1 size", matrix1Size },
           { "matrix2 size", matrix2Size } } ) );
}

template< typename Real, typename Index >
void
setMetadata( TNL::Benchmarks::Benchmark& benchmark, const std::string& algorithm, const std::string& matrixSize )
{
   benchmark.setMetadataColumns(
      TNL::Benchmarks::Benchmark::MetadataColumns(
         { { "index type", TNL::getType< Index >() },
           { "real type", TNL::getType< Real >() },
           { "algorithm", algorithm },
           { "matrix size", matrixSize } } ) );
}

template< typename Real, typename Index >
void
fillHostMatrix( TNL::Matrices::DenseMatrix< Real, Devices::Host, Index >& matrix, bool linearFill, Real baseValue )
{
   const Index rows = matrix.getRows();
   const Index cols = matrix.getColumns();
   const Real h_x = 1.0 / 100;
   const Real h_y = 1.0 / 100;
   for( Index i = 0; i < rows; i++ ) {
      for( Index j = 0; j < cols; j++ ) {
         Real value;
         if( linearFill )
            value = baseValue + i * 2;
         else
            value = std::sin( 2 * M_PI * h_x * i ) + std::cos( 2 * M_PI * h_y * i );
         matrix.setElement( i, j, value );
      }
   }
}

template< typename Matrix >
void
fillGpuMatrix( Matrix& matrix, bool linearFill, typename Matrix::RealType baseValue )
{
   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   auto view = matrix.getView();
   const Index rows = matrix.getRows();
   const Index cols = matrix.getColumns();
   const Real h_x = 1.0 / 100;
   const Real h_y = 1.0 / 100;

   auto fill = [ = ] __cuda_callable__( Index colIdx ) mutable
   {
      for( Index row = 0; row < rows; row++ ) {
         Real value;
         if( linearFill )
            value = baseValue + row * 2;
         else
            value = std::sin( 2 * M_PI * h_x * row ) + std::cos( 2 * M_PI * h_y * row );
         view.setElement( row, colIdx, value );
      }
   };
   TNL::Algorithms::parallelFor< Devices::GPU >( 0, cols, fill );
}

inline std::string
transposeSuffix( TransposeState transposeA, TransposeState transposeB )
{
   std::string suffix;
   if( transposeA == TransposeState::Transpose )
      suffix += 'A';
   if( transposeB == TransposeState::Transpose )
      suffix += 'B';
   return suffix;
}

template< typename Matrix >
std::string
sizeString( const Matrix& matrix, TransposeState ts )
{
   if( ts == TransposeState::Transpose )
      return std::to_string( matrix.getColumns() ) + "x" + std::to_string( matrix.getRows() );
   return std::to_string( matrix.getRows() ) + "x" + std::to_string( matrix.getColumns() );
}

}  // namespace TNL::Benchmarks::DenseMatrices

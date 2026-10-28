// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/SparseMatrix.h>

#include "LaplaceProblem.h"
#include "Utils.h"

namespace TNL::Benchmarks::MatrixAssembly {

//! \brief Allocates the matrix: sets the dimensions and the row capacities.
template< typename Matrix, typename Problem >
void
allocateTNLMatrix( Matrix& matrix, const Problem& problem )
{
   using Device = typename Matrix::DeviceType;
   using Index = typename Matrix::IndexType;

   const Index rows = problem.getRows();
   matrix.setDimensions( rows, rows );
   Containers::Vector< Index, Device, Index > rowCapacities( rows );
   auto capacitiesView = rowCapacities.getView();
   Algorithms::parallelFor< Device >(
      0,
      rows,
      [ = ] __cuda_callable__( Index row ) mutable
      {
         capacitiesView[ row ] = problem.getRowLength( row );
      } );
   matrix.setRowCapacities( rowCapacities );
}

//! \brief Fills an allocated matrix with the non-zero elements in parallel (one thread per row).
template< typename Matrix, typename Problem >
void
fillTNLMatrix( Matrix& matrix, const Problem& problem )
{
   using Index = typename Matrix::IndexType;
   using Real = typename Matrix::RealType;
   using RowView = typename Matrix::RowView;

   matrix.forAllRows(
      [ = ] __cuda_callable__( RowView& row ) mutable
      {
         problem.forRowElements(
            row.getRowIndex(),
            [ & ]( Index localIdx, Index column, Real value )
            {
               row.setElement( localIdx, column, value );
            } );
      } );
}

//! \brief Copies a TNL sparse matrix (in any format, on any device) into host CSR arrays.
template< typename Matrix >
void
getTNLMatrixCSR(
   const Matrix& matrix,
   std::vector< typename Matrix::IndexType >& rowPointers,
   std::vector< typename Matrix::IndexType >& columnIndexes,
   std::vector< typename Matrix::RealType >& values )
{
   using Index = typename Matrix::IndexType;
   using HostMatrix = typename Matrix::template Self< typename Matrix::RealType, Devices::Host >;

   HostMatrix hostMatrix;
   hostMatrix = matrix;
   const Index rows = hostMatrix.getRows();
   rowPointers.assign( 1, 0 );
   columnIndexes.clear();
   values.clear();
   for( Index rowIdx = 0; rowIdx < rows; rowIdx++ ) {
      const auto row = hostMatrix.getRow( rowIdx );
      for( Index i = 0; i < row.getSize(); i++ ) {
         const Index column = row.getColumnIndex( i );
         if( column == Matrices::paddingIndex< Index > )
            continue;
         columnIndexes.push_back( column );
         values.push_back( row.getValue( i ) );
      }
      rowPointers.push_back( columnIndexes.size() );
   }
}

/**
 * \brief Benchmarks the assembly of a TNL sparse matrix with the given segments (format).
 *
 * Measured operations:
 *
 * - `allocation`: setting the dimensions and the row capacities,
 * - `filling`: writing the non-zero elements into an allocated matrix,
 * - `assembly`: allocation + filling.
 *
 * For GPU devices, the assembly on the host followed by the transfer to the GPU is measured in
 * addition (see \ref timeHostAssemblyAndTransfer). The transfer uses the assignment operator of
 * the sparse matrix. The operation `array transfer` measures copying only the arrays of values
 * and column indexes, which is the lower bound of the transfer time.
 */
template< typename Real, typename Device, typename Index, template< typename, typename, typename > class Segments >
void
benchmarkTNL( Benchmark& benchmark, const LaplaceProblem< Real, Index >& problem, const std::string& format, bool verify )
{
   using Matrix = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Segments >;

   benchmark.setMetadataElement( { "format", format } );
   Matrix matrix;

   auto resetMatrix = [ & ]()
   {
      matrix.reset();
   };
   auto allocate = [ & ]()
   {
      allocateTNLMatrix( matrix, problem );
   };
   auto fill = [ & ]()
   {
      fillTNLMatrix( matrix, problem );
   };
   auto assemble = [ & ]()
   {
      allocateTNLMatrix( matrix, problem );
      fillTNLMatrix( matrix, problem );
   };

   timeOperation< Device >( benchmark, "allocation", "TNL", resetMatrix, allocate );
   // the matrix is allocated after the last allocation loop
   timeOperation< Device >( benchmark, "filling", "TNL", noReset, fill );
   timeOperation< Device >( benchmark, "assembly", "TNL", resetMatrix, assemble );

   if( verify ) {
      std::vector< Index > rowPointers;
      std::vector< Index > columnIndexes;
      std::vector< Real > values;
      getTNLMatrixCSR( matrix, rowPointers, columnIndexes, values );
      reportVerification( benchmark, "TNL", problem.compare( rowPointers, columnIndexes, values ) );
   }

   if constexpr( ! std::is_same_v< Device, Devices::Host > ) {
      using HostMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index, Matrices::GeneralMatrix, Segments >;
      HostMatrix hostMatrix;

      auto resetAll = [ & ]()
      {
         hostMatrix.reset();
         matrix.reset();
      };
      auto hostAssemble = [ & ]()
      {
         allocateTNLMatrix( hostMatrix, problem );
         fillTNLMatrix( hostMatrix, problem );
      };
      auto transfer = [ & ]()
      {
         matrix = hostMatrix;
      };
      timeHostAssemblyAndTransfer< Device >( benchmark, "TNL", resetAll, resetMatrix, hostAssemble, transfer );

      if( verify ) {
         std::vector< Index > rowPointers;
         std::vector< Index > columnIndexes;
         std::vector< Real > values;
         getTNLMatrixCSR( matrix, rowPointers, columnIndexes, values );
         reportVerification( benchmark, "TNL", problem.compare( rowPointers, columnIndexes, values ) );
      }

      // lower bound of the transfer: copying only the arrays of values and column indexes
      // (the host matrix is assembled after the last loop of the previous measurement)
      Containers::Vector< Real, Device, Index > valuesCopy;
      Containers::Vector< Index, Device, Index > columnIndexesCopy;
      auto resetArrays = [ & ]()
      {
         valuesCopy.reset();
         columnIndexesCopy.reset();
      };
      auto transferArrays = [ & ]()
      {
         valuesCopy = hostMatrix.getValues();
         columnIndexesCopy = hostMatrix.getColumnIndexes();
      };
      timeOperation< Device >( benchmark, "array transfer", "TNL", resetArrays, transferArrays );
   }
}

}  // namespace TNL::Benchmarks::MatrixAssembly

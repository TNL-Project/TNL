// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixBase.h>

#ifdef HAVE_BLAS

   #include <cblas.h>

namespace TNL::Benchmarks::DenseMatrices {

template< typename DenseMatrix >
void
matrixMultiplicationBLAS( const DenseMatrix& matrix1, const DenseMatrix& matrix2, DenseMatrix& resultMatrix )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::Host >, "This function is specialized for Host device only." );

   IndexType n = matrix2.getColumns();
   IndexType k = matrix1.getColumns();
   IndexType m = matrix1.getRows();

   auto organization = matrix1.getOrganization();

   if constexpr( std::is_same_v< RealType, float > ) {
      cblas_sgemm(
         organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
         CblasNoTrans,
         CblasNoTrans,
         m,
         n,
         k,
         1.0F,
         matrix1.getValues().getData(),
         k,
         matrix2.getValues().getData(),
         n,
         0.0F,
         resultMatrix.getValues().getData(),
         organization == TNL::Algorithms::Segments::RowMajorOrder ? n : m );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      cblas_dgemm(
         organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
         CblasNoTrans,
         CblasNoTrans,
         m,
         n,
         k,
         1.0,
         matrix1.getValues().getData(),
         k,
         matrix2.getValues().getData(),
         n,
         0.0,
         resultMatrix.getValues().getData(),
         organization == TNL::Algorithms::Segments::RowMajorOrder ? n : m );
   }
}

template< typename DenseMatrix, typename Vector >
void
matrixVectorProductBLAS( const DenseMatrix& matrix, const Vector& inVector, Vector& outVector )
{
   using RealType = typename DenseMatrix::RealType;
   using IndexType = typename DenseMatrix::IndexType;
   using Device = typename DenseMatrix::DeviceType;

   static_assert( std::is_same_v< Device, TNL::Devices::Host >, "This function is specialized for Host device only." );

   IndexType rows = matrix.getRows();
   IndexType cols = matrix.getColumns();

   auto organization = matrix.getOrganization();

   if constexpr( std::is_same_v< RealType, float > ) {
      cblas_sgemv(
         organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
         CblasNoTrans,
         rows,
         cols,
         1.0F,
         matrix.getValues().getData(),
         organization == TNL::Algorithms::Segments::RowMajorOrder ? cols : rows,
         inVector.getData(),
         1,
         0.0F,
         outVector.getData(),
         1 );
   }
   else if constexpr( std::is_same_v< RealType, double > ) {
      cblas_dgemv(
         organization == TNL::Algorithms::Segments::RowMajorOrder ? CblasRowMajor : CblasColMajor,
         CblasNoTrans,
         rows,
         cols,
         1.0,
         matrix.getValues().getData(),
         organization == TNL::Algorithms::Segments::RowMajorOrder ? cols : rows,
         inVector.getData(),
         1,
         0.0,
         outVector.getData(),
         1 );
   }
}

}  // namespace TNL::Benchmarks::DenseMatrices

#endif  //HAVE_BLAS

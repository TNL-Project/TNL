// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>

#include "DenseMatrix.h"
#include "SparseOperations.h"
#include "DenseOperations.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix( const RealAllocatorType& allocator )
: values( allocator )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix( const DenseMatrix& matrix )
: values( matrix.values ),
  segments( matrix.segments )
{
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), segments.getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix( Index rows,
                                                                              Index columns,
                                                                              const RealAllocatorType& allocator )
: values( allocator )
{
   this->setDimensions( rows, columns );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Value >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix(
   std::initializer_list< std::initializer_list< Value > > data,
   const RealAllocatorType& allocator )
: values( allocator )
{
   this->setElements( data );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Value >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setElements(
   std::initializer_list< std::initializer_list< Value > > data )
{
   Index rows = data.size();
   Index columns = 0;
   for( auto row : data )
      columns = max( columns, row.size() );
   this->setDimensions( rows, columns );
   if constexpr( std::is_same_v< Device, Devices::Cuda > ) {
      DenseMatrix< Real, Devices::Host, Index > hostDense( rows, columns );
      Index rowIdx = 0;
      for( auto row : data ) {
         Index columnIdx = 0;
         for( auto element : row )
            hostDense.setElement( rowIdx, columnIdx++, element );
         rowIdx++;
      }
      *this = hostDense;
   }
   else {
      Index rowIdx = 0;
      for( auto row : data ) {
         Index columnIdx = 0;
         for( auto element : row )
            this->setElement( rowIdx, columnIdx++, element );
         rowIdx++;
      }
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename MapIndex, typename MapValue >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setElements(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   MatrixElementsEncoding encoding )
{
   if constexpr( ! std::is_same_v< Device, Devices::Host > && ! std::is_same_v< Device, Devices::Sequential > ) {
      DenseMatrix< Real, Devices::Host, Index, Organization > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setElements( map );
      *this = hostMatrix;
   }
   else {
      for( const auto& [ coordinates, value ] : map ) {
         const auto& [ rowIdx, columnIdx ] = coordinates;
         if( rowIdx >= this->getRows() )
            throw std::logic_error( "Wrong row index " + std::to_string( rowIdx ) + " in the input data structure." );
         if( columnIdx >= this->getColumns() )
            throw std::logic_error( "Wrong column index " + std::to_string( columnIdx ) + " in the input data structure." );
         if( encoding == MatrixElementsEncoding::SymmetricMixed ) {
            auto query = map.find( { columnIdx, rowIdx } );
            if( query != map.end() && query->second != value )
               throw std::logic_error( "The input data are supposed to be symmetric (matrix elements encoding equals "
                                       "SymmetricMixed) but it is not. The matrix elements at position ("
                                       + std::to_string( rowIdx ) + ", " + std::to_string( columnIdx ) + ") do not match." );
         }

         this->setElement( rowIdx, columnIdx, value );
         if( ( encoding == MatrixElementsEncoding::SymmetricMixed || encoding == MatrixElementsEncoding::SymmetricLower
               || encoding == MatrixElementsEncoding::SymmetricUpper )
             && rowIdx != columnIdx )
            this->setElement( columnIdx, rowIdx, value );
      }
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getView() -> ViewType
{
   return { this->getRows(), this->getColumns(), this->getValues().getView() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getConstView() const -> ConstViewType
{
   return { this->getRows(), this->getColumns(), this->getValues().getConstView() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setDimensions( Index rows, Index columns )
{
   this->segments.setSegmentsSizes( rows, columns );
   this->values.setSize( this->segments.getStorageSize() );
   this->values = 0.0;
   // update the base
   Base::bind( rows, columns, values.getView(), segments.getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix_ >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setLike( const Matrix_& matrix )
{
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RowCapacitiesVector >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ( rowCapacities.getSize(), this->getRows(), "" );
   TNL_ASSERT_LE( max( rowCapacities ), this->getColumns(), "" );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::reset()
{
   this->values.reset();
   this->segments.reset();
   // update the base
   Base::bind( 0, 0, values.getView(), segments.getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix1, typename Matrix2, int tileDim >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getMatrixProduct( const Matrix1& matrix1,
                                                                                   const Matrix2& matrix2,
                                                                                   Real matrixMultiplicator,
                                                                                   TransposeState transposeA,
                                                                                   TransposeState transposeB )
{
   TNL::Matrices::getMatrixProduct( *this, matrix1, matrix2, matrixMultiplicator, transposeA, transposeB );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix, int tileDim >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getTransposition( const Matrix& matrix,
                                                                                   Real matrixMultiplicator )
{
   TNL::Matrices::getTransposition( *this, matrix, matrixMultiplicator );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< int tileDim >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getInPlaceTransposition( Real matrixMultiplicator )
{
   TNL::Matrices::getInPlaceTransposition( *this, matrixMultiplicator );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix )
{
   return this->operator=( matrix.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   DenseMatrix< Real, Device, Index, Organization, RealAllocator >&& matrix ) noexcept( false )
{
   this->values = std::move( matrix.values );
   this->segments = std::move( matrix.segments );
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), segments.getView() );
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal, typename RHSDevice, typename RHSIndex, typename RHSRealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrix< RHSReal, RHSDevice, RHSIndex, Organization, RHSRealAllocator >& matrix )
{
   return this->operator=( matrix.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal, typename RHSDevice, typename RHSIndex >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, Organization >& matrix )
{
   this->setLike( matrix );
   this->values = matrix.getValues();
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal,
          typename RHSDevice,
          typename RHSIndex,
          ElementsOrganization RHSOrganization,
          typename RHSRealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSOrganization, RHSRealAllocator >& matrix )
{
   return this->operator=( matrix.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal, typename RHSDevice, typename RHSIndex, ElementsOrganization RHSOrganization >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, RHSOrganization >& matrix )
{
   copyDenseToDenseMatrix( *this, matrix );
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSMatrix >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=( const RHSMatrix& matrix )
{
   copySparseToDenseMatrix( *this, matrix );
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::save( const String& fileName ) const
{
   File( fileName, std::ios_base::out ) << *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::load( const String& fileName )
{
   File( fileName, std::ios_base::in ) >> *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
File&
operator>>( File& file, DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix )
{
   const std::string type = getObjectType( file );
   if( type != matrix.getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "object type does not match (expected " + matrix.getSerializationType()
                                                     + ", found " + type + ")." );
   std::size_t rows = 0;
   std::size_t columns = 0;
   file.load( &rows );
   file.load( &columns );
   // setDimensions initializes the internal segments attribute
   matrix.setDimensions( rows, columns );
   file >> matrix.getValues();
   return file;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
File&
operator>>( File&& file, DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file >> matrix;
}

}  // namespace TNL::Matrices

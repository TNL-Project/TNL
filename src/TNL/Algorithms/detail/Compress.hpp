// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/scan.h>
#include "../compress.h"  // IWYU pragma: keep

namespace TNL::Algorithms::detail {

// Writes indices of the marks into outputVector, given the inclusive prefix sum
// of the marks in scannedMarks.
template< typename ScannedMarks, typename OutputVector, typename Index >
Index
compress_scatter( ScannedMarks&& scannedMarks, OutputVector& outputVector, Index shift )
{
   TNL_ASSERT_GE( shift, 0, "" );
   auto outputSize = scannedMarks.getElement( scannedMarks.getSize() - 1 );
   if( outputSize > outputVector.getSize() )
      outputVector.setSize( outputSize );
   auto outputView = outputVector.getView();
   auto marksView = scannedMarks.getView();
   auto f = [ = ] __cuda_callable__( const Index idx, const Index value ) mutable
   {
      if( idx == 0 ) {
         if( value == 1 )
            outputView[ 0 ] = idx + shift;
      }
      else if( value - marksView[ idx - 1 ] == 1 ) {
         outputView[ value - 1 ] = idx + shift;
      }
   };
   marksView.forAllElements( f );
   return outputSize;
}

template< typename MarksVector, typename OutputVector, typename Index >
Index
compress_impl( MarksVector&& marksVector, OutputVector& outputVector, Index shift )
{
   TNL_ASSERT_GE( shift, 0, "" );
   Algorithms::inplaceInclusiveScan( marksVector );
   return compress_scatter( std::forward< MarksVector >( marksVector ), outputVector, shift );
}

template< typename BeginIndex, typename EndIndex, typename MarksFunction, typename OutputVector >
auto
compress( BeginIndex begin, EndIndex end, MarksFunction&& marksFunction, OutputVector& outputVector ) ->
   typename OutputVector::IndexType
{
   TNL_ASSERT_GE( begin, 0, "" );
   TNL_ASSERT_GE( end, begin, "" );

   using Device = typename OutputVector::DeviceType;
   using Index = typename OutputVector::IndexType;
   OutputVector marksVector( end - begin );
   auto marksView = marksVector.getView();
   Algorithms::parallelFor< Device >(
      begin,
      end,
      [ = ] __cuda_callable__( Index idx ) mutable
      {
         marksView[ idx - begin ] = marksFunction( idx );
      } );
   return compress_impl( std::forward< OutputVector >( marksVector ), outputVector, begin );
}

template< typename OutputVector, typename BeginIndex, typename EndIndex, typename MarksFunction >
OutputVector
compress( BeginIndex begin, EndIndex end, MarksFunction&& marksFunction )
{
   TNL_ASSERT_GE( begin, 0, "" );
   TNL_ASSERT_GE( end, begin, "" );

   OutputVector outputVector;
   compress( begin, end, std::forward< MarksFunction >( marksFunction ), outputVector );
   return outputVector;
}

template< typename MarksVector, typename OutputVector, typename BeginIndex, typename EndIndex >
auto
compressVector( MarksVector& marksVector, OutputVector& outputVector, BeginIndex begin, EndIndex end ) ->
   typename OutputVector::IndexType
{
   TNL_ASSERT_GE( begin, 0, "" );
   TNL_ASSERT_GE( end, begin, "" );
   if( end == 0 )
      end = marksVector.getSize();

   auto marksView = marksVector.getView( begin, end );
   return compress_impl( marksView, outputVector, begin );
}

template< typename MarksVector, typename OutputVector, typename AuxVector, typename BeginIndex, typename EndIndex >
auto
compressVector(
   const MarksVector& marksVector,
   OutputVector& outputVector,
   AuxVector& auxVector,
   BeginIndex begin,
   EndIndex end ) -> typename OutputVector::IndexType
{
   TNL_ASSERT_GE( begin, 0, "" );
   TNL_ASSERT_GE( end, begin, "" );
   if( end == 0 )
      end = marksVector.getSize();

   using Index = typename OutputVector::IndexType;
   const Index size = end - begin;
   if( size == 0 )
      return 0;
   if( auxVector.getSize() < size )
      auxVector.setSize( size );
   auto auxView = auxVector.getView( 0, size );
   Algorithms::inclusiveScan( marksVector, auxView, begin, end, 0 );
   return compress_scatter( auxView, outputVector, static_cast< Index >( begin ) );
}

template< typename MarksVector, typename OutputVector, typename BeginIndex, typename EndIndex >
OutputVector
compressVector( MarksVector& marksVector, BeginIndex begin, EndIndex end )
{
   TNL_ASSERT_GE( begin, 0, "" );
   TNL_ASSERT_GE( end, begin, "" );
   if( end == 0 )
      end = marksVector.getSize();

   OutputVector outputVector;
   compressVector( marksVector, outputVector, begin, end );
   return outputVector;
}

}  // namespace TNL::Algorithms::detail

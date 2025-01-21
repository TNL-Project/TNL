// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/scan.h>
#include "../compress.h"

namespace TNL::Algorithms::detail {

template< typename MarksVector, typename OutputVector, typename Index >
Index
compress_impl( MarksVector&& marksVector, OutputVector& outputVector, Index shift )
{
   TNL_ASSERT_GE( shift, 0, "" );
   Algorithms::inplaceInclusiveScan( marksVector );
   auto outputSize = marksVector.getElement( marksVector.getSize() - 1 );
   if( outputSize > outputVector.getSize() )
      outputVector.setSize( outputSize );
   auto outputView = outputVector.getView();
   auto marksView = marksVector.getView();
   auto f = [ = ] __cuda_callable__( const Index idx, const Index value ) mutable
   {
      if( idx == 0 ) {
         if( value == 1 )
            outputView[ 0 ] = idx + shift;
      }
      else if( value - marksView[ idx - 1 ] == 1 )
         outputView[ value - 1 ] = idx + shift;
   };
   marksView.forAllElements( f );
   return outputSize;
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
   Algorithms::parallelFor< Device >( begin,
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

}  //namespace TNL::Algorithms::detail

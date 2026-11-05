// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>

#include "../uncompress.h"

namespace TNL::Algorithms::detail {

/**
 * Core implementation of uncompress and uncompressUnchecked.
 *
 * If \p maskSize is zero and the index vector is non-empty, the required mask
 * size is derived automatically as \c max(indexVector) + 1.
 *
 * If \p Checked is true, both validation (runtime throw on out-of-range index)
 * and the actual write are performed in a single parallel reduction pass to
 * avoid redundant memory traversal. If \p Checked is false, the indices are
 * written by a plain parallelFor without validation (only debug assertions).
 *
 * Only the indices in the range [begin, end) of the index vector are used.
 * If \p end is zero, it is set to the size of the index vector.
 *
 * If the index vector and the mask live on different devices, the indexes
 * (only those in the range [begin, end)) are copied to the device of the mask first.
 */
template< bool Checked, typename IndexVector, typename MaskVector >
void
uncompress_impl(
   const IndexVector& indexVector,
   MaskVector& maskVector,
   typename IndexVector::IndexType maskSize,
   typename IndexVector::IndexType begin,
   typename IndexVector::IndexType end )
{
   using Device = typename IndexVector::DeviceType;
   using Index = typename IndexVector::IndexType;
   using MaskValue = typename MaskVector::ValueType;
   using MaskDevice = typename MaskVector::DeviceType;

   if( end == 0 )
      end = indexVector.getSize();
   TNL_ASSERT_GE( begin, 0, "" );
   TNL_ASSERT_GE( end, begin, "" );
   TNL_ASSERT_LE( end, indexVector.getSize(), "" );

   if constexpr( ! std::is_same_v< Device, MaskDevice > ) {
      Containers::Vector< typename IndexVector::ValueType, MaskDevice, Index > indexes;
      indexes = indexVector.getConstView( begin, end );
      uncompress_impl< Checked >( indexes, maskVector, maskSize, Index( 0 ), indexes.getSize() );
      return;
   }

   const Index n = end - begin;

   // Auto-detect maskSize: find the maximum index and add 1.
   if( maskSize == 0 ) {
      if( n == 0 ) {
         maskVector.setSize( 0 );
         return;
      }
      maskSize = max( indexVector.getConstView( begin, end ) ) + Index( 1 );
   }

   maskVector.setSize( maskSize );
   maskVector = MaskValue( 0 );

   if( n == 0 )
      return;

   const auto indexView = indexVector.getConstView();
   auto maskView = maskVector.getView();

   if constexpr( ! Checked ) {
      Algorithms::parallelFor< Device >(
         begin,
         end,
         [ = ] __cuda_callable__( Index k ) mutable
         {
            const Index idx = indexView[ k ];
            TNL_ASSERT_GE( idx, Index( 0 ), "uncompressUnchecked: negative index" );
            TNL_ASSERT_LT( idx, maskSize, "uncompressUnchecked: index out of range" );
            maskView[ idx ] = MaskValue( 1 );
         } );
      return;
   }

   // Validate all indices and set the mask in a single pass.
   // For each valid index the corresponding mask entry is set to 1.
   // An invalid index causes the lambda to return false, which the
   // LogicalAnd reduction propagates so we can detect and report the error.
   const bool valid = Algorithms::reduce< Device >(
      begin,
      end,
      [ = ] __cuda_callable__( Index k ) mutable -> bool
      {
         const Index idx = indexView[ k ];
         if( idx >= Index( 0 ) && idx < maskSize ) {
            maskView[ idx ] = MaskValue( 1 );
            return true;
         }
         return false;
      },
      TNL::LogicalAnd{} );

   if( ! valid )
      throw std::invalid_argument( "uncompress: index out of range — all indices must be in [0, maskSize)." );
}

}  // namespace TNL::Algorithms::detail

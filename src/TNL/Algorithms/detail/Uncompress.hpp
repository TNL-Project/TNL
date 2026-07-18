// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Functional.h>

#include "../uncompress.h"

namespace TNL::Algorithms::detail {

/**
 * Core implementation of uncompress.
 *
 * If \p maskSize is zero and the index vector is non-empty, the required mask
 * size is derived automatically as \c max(indexVector) + 1.
 *
 * Both validation (runtime throw on out-of-range index) and the actual write
 * are performed in a single parallel reduction pass to avoid redundant
 * memory traversal.
 */
template< typename IndexVector, typename MaskVector >
void
uncompress_impl( const IndexVector& indexVector, MaskVector& maskVector, typename IndexVector::IndexType maskSize )
{
   using Device = typename IndexVector::DeviceType;
   using Index = typename IndexVector::IndexType;
   using MaskValue = typename MaskVector::ValueType;

   const Index n = static_cast< Index >( indexVector.getSize() );

   // Auto-detect maskSize: find the maximum index and add 1.
   if( maskSize == 0 ) {
      if( n == 0 ) {
         maskVector.setSize( 0 );
         return;
      }
      const auto indexView = indexVector.getConstView();
      maskSize = Algorithms::reduce< Device >(
                    Index( 0 ),
                    n,
                    [ = ] __cuda_callable__( Index k )
                    {
                       return indexView[ k ];
                    },
                    TNL::Max{} ) +
                 Index( 1 );
   }

   maskVector.setSize( maskSize );
   maskVector = MaskValue( 0 );

   if( n == 0 )
      return;

   const auto indexView = indexVector.getConstView();
   auto maskView = maskVector.getView();

   // Validate all indices and set the mask in a single pass.
   // For each valid index the corresponding mask entry is set to 1.
   // An invalid index causes the lambda to return false, which the
   // LogicalAnd reduction propagates so we can detect and report the error.
   const bool valid = Algorithms::reduce< Device >(
      Index( 0 ),
      n,
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

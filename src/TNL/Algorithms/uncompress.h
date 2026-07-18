// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/TypeTraits.h>

#include "detail/Uncompress.hpp"

namespace TNL::Algorithms {

/**
 * \brief Converts a list of indices into a boolean mask (inverse of \ref compress).
 *
 * Given a vector of indices \p indexVector and a target mask size \p maskSize,
 * returns a new vector of size \p maskSize where every position listed in
 * \p indexVector is set to 1 and all remaining positions are set to 0.
 *
 * If \p maskSize is 0 (the default), the required size is derived automatically
 * as `max(indexVector) + 1`. Passing an empty \p indexVector with \p maskSize == 0
 * returns an empty mask.
 *
 * All indices must satisfy `0 <= idx < maskSize`; a \c std::invalid_argument
 * exception is thrown otherwise.
 *
 * \note \c uncompress is the exact inverse of \ref compress:
 * \code
 *   auto indices = compress< Vector >( mask );  // mask → indices
 *   auto mask2   = uncompress( indices, mask.getSize() );  // indices → mask
 *   // mask2 == mask
 * \endcode
 *
 * \tparam IndexVector Type of the input index vector.
 * \tparam MaskVector  Type of the returned mask vector (defaults to IndexVector).
 * \param indexVector  Vector of indices to mark as active (1).
 * \param maskSize     Size of the output mask. 0 means auto-detect from the input.
 * \return A new vector of size \p maskSize with 1 at each listed index.
 *
 * \par Example
 * \include Algorithms/uncompressExample.cpp
 * \par Output
 * \include uncompressExample.out
 */
template< typename IndexVector,
          typename MaskVector = IndexVector,
          typename std::enable_if_t< IsArrayType< std::decay_t< IndexVector > >::value, bool > = true >
MaskVector
uncompress( const IndexVector& indexVector, typename IndexVector::IndexType maskSize = 0 )
{
   MaskVector maskVector;
   detail::uncompress_impl( indexVector, maskVector, maskSize );
   return maskVector;
}

/**
 * \brief Converts a list of indices into a boolean mask, filling an existing vector.
 *
 * Equivalent to the returning overload, but writes the result into the provided
 * \p maskVector instead of allocating a new one. \p maskVector is resized to
 * \p maskSize (or to `max(indexVector) + 1` if \p maskSize == 0) and zeroed
 * before filling.
 *
 * \tparam IndexVector Type of the input index vector.
 * \tparam MaskVector  Type of the output mask vector.
 * \param indexVector  Vector of indices to mark as active (1).
 * \param maskVector   Output vector to fill.
 * \param maskSize     Size of the output mask. 0 means auto-detect from the input.
 *
 * \par Example
 * \include Algorithms/uncompressExample.cpp
 * \par Output
 * \include uncompressExample.out
 */
template< typename IndexVector,
          typename MaskVector,
          typename std::enable_if_t< IsArrayType< std::decay_t< IndexVector > >::value
                                        && IsArrayType< std::decay_t< MaskVector > >::value,
                                     bool > = true >
void
uncompress( const IndexVector& indexVector, MaskVector& maskVector, typename IndexVector::IndexType maskSize = 0 )
{
   detail::uncompress_impl( indexVector, maskVector, maskSize );
}

}  // namespace TNL::Algorithms

// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/TypeTraits.h>

#include "detail/Uncompress.hpp"

namespace TNL::Algorithms {

// clang-format off
/**
 * \page UncompressOverview Overview of Uncompress Functions
 *
 * \tableofcontents
 *
 * This page gives an overview of the \ref TNL::Algorithms::uncompress and
 * \ref TNL::Algorithms::uncompressUnchecked functions, the differences between them
 * and guidance for choosing the right one.
 *
 * \section UncompressWhatIsUncompress What is Uncompress?
 *
 * **Uncompress** converts a list of indices into a mask: a vector where the
 * positions listed in the index vector are set to 1 and all other positions to 0.
 * It is the inverse of \ref TNL::Algorithms::compress (see \ref CompressOverview).
 * For example, with the mask size 11:
 *
 * ```
 * indices: 0  5  7  8
 * index:   0  1  2  3  4  5  6  7  8  9 10
 * mask:    1  0  0  0  0  1  0  1  1  0  0
 * ```
 *
 * All variants work in parallel on any device. They set the mask to 0 and then
 * write 1 to each listed position.
 *
 * \section UncompressFunctions Overview of the Variants
 *
 * The variants differ in two ways:
 * - **how the mask is returned**: as a new vector, or written into a vector the
 *   caller provides,
 * - **whether the indices are checked**: \ref TNL::Algorithms::uncompress checks
 *   that every index is in `[0, maskSize)` and throws an exception otherwise;
 *   \ref TNL::Algorithms::uncompressUnchecked does not check them.
 *
 * | Function                                                         | Mask            | Indices checked                      |
 * |------------------------------------------------------------------|-----------------|--------------------------------------|
 * | `uncompress( indices, maskSize, begin, end )`                    | returned vector | yes, throws `std::invalid_argument`  |
 * | `uncompress( indices, mask, maskSize, begin, end )`              | `mask`          | yes, throws `std::invalid_argument`  |
 * | `uncompressUnchecked( indices, maskSize, begin, end )`           | returned vector | no (only assertions in debug builds) |
 * | `uncompressUnchecked( indices, mask, maskSize, begin, end )`     | `mask`          | no (only assertions in debug builds) |
 *
 * \section UncompressCommonRules Common Rules
 *
 * - Only the indices in the range `[begin, end)` of the index vector are used.
 *   `end == 0` (the default) means the end of the index vector, so an empty range
 *   ending at 0 cannot be expressed; handle an empty range on the caller side.
 * - If `maskSize == 0` (the default), the mask size is set to the largest used
 *   index plus one. This needs an extra reduction, so pass the mask size
 *   explicitly whenever it is known.
 * - The mask is always set to 0 first, so the result does not depend on its
 *   previous content. A mask that already has the size `maskSize` is not
 *   reallocated.
 * - Duplicate indices are allowed.
 * - If the index vector and the mask are on different devices, the used indices
 *   are copied to the device of the mask first.
 *
 * \section UncompressChoosing Choosing a Variant
 *
 * - Use **`uncompress`** when the indices come from outside, e.g. from user input
 *   or a file, and an invalid index must be reported.
 * - Use **`uncompressUnchecked`** when the indices are valid by construction, e.g.
 *   when they were produced by \ref TNL::Algorithms::compress, and the function is
 *   called repeatedly. The check in `uncompress` is a reduction whose result must
 *   be read on the host: on GPUs this means a synchronization in every call, on the
 *   host an OpenMP reduction. `uncompressUnchecked` uses a plain `parallelFor` instead.
 * - An invalid index passed to `uncompressUnchecked` causes undefined behavior
 *   (an out-of-bounds write) in release builds.
 */
// clang-format on

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
 * Only the indices in the range `[begin, end)` of \p indexVector are used.
 * If \p end is 0 (the default), it is set to the size of \p indexVector, so
 * `[begin, end)` cannot describe an empty range ending at 0; handle an empty
 * range on the caller side.
 *
 * All used indices must satisfy `0 <= idx < maskSize`; a \c std::invalid_argument
 * exception is thrown otherwise. If the indices are valid by construction, use
 * \ref TNL::Algorithms::uncompressUnchecked, which avoids the cost of the check.
 * See \ref UncompressOverview for a comparison of the variants.
 *
 * \note \c uncompress is the exact inverse of \ref compress "compress":
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
 * \param begin        Beginning of the range of \p indexVector to use.
 * \param end          End of the range of \p indexVector to use. 0 means the end of \p indexVector.
 * \return A new vector of size \p maskSize with 1 at each listed index.
 *
 * \par Example
 * \include Algorithms/uncompressExample.cpp
 * \par Output
 * \include uncompressExample.out
 */
template<
   typename IndexVector,
   typename MaskVector = IndexVector,
   typename std::enable_if_t< IsArrayType< std::decay_t< IndexVector > >::value, bool > = true >
MaskVector
uncompress(
   const IndexVector& indexVector,
   typename IndexVector::IndexType maskSize = 0,
   typename IndexVector::IndexType begin = 0,
   typename IndexVector::IndexType end = 0 )
{
   MaskVector maskVector;
   detail::uncompress_impl< true >( indexVector, maskVector, maskSize, begin, end );
   return maskVector;
}

/**
 * \brief Converts a list of indices into a boolean mask, filling an existing vector.
 *
 * Equivalent to the returning overload, but writes the result into the provided
 * \p maskVector instead of allocating a new one. \p maskVector is resized to
 * \p maskSize (or to `max(indexVector) + 1` if \p maskSize == 0) and zeroed
 * before filling. If \p maskVector already has the size \p maskSize, no
 * reallocation is performed.
 *
 * \tparam IndexVector Type of the input index vector.
 * \tparam MaskVector  Type of the output mask vector.
 * \param indexVector  Vector of indices to mark as active (1).
 * \param maskVector   Output vector to fill.
 * \param maskSize     Size of the output mask. 0 means auto-detect from the input.
 * \param begin        Beginning of the range of \p indexVector to use.
 * \param end          End of the range of \p indexVector to use. 0 means the end of \p indexVector.
 *
 * \par Example
 * \include Algorithms/uncompressExample.cpp
 * \par Output
 * \include uncompressExample.out
 */
template<
   typename IndexVector,
   typename MaskVector,
   typename std::enable_if_t<
      IsArrayType< std::decay_t< IndexVector > >::value && IsArrayType< std::decay_t< MaskVector > >::value,
      bool > = true >
void
uncompress(
   const IndexVector& indexVector,
   MaskVector& maskVector,
   typename IndexVector::IndexType maskSize = 0,
   typename IndexVector::IndexType begin = 0,
   typename IndexVector::IndexType end = 0 )
{
   detail::uncompress_impl< true >( indexVector, maskVector, maskSize, begin, end );
}

/**
 * \brief Converts a list of indices into a boolean mask without checking the indices.
 *
 * Same as \ref TNL::Algorithms::uncompress, but the indices are not checked to be in
 * `[0, maskSize)`. The mask is written by a plain parallel for loop, which avoids the
 * reduction (and on GPUs the synchronization) needed for the check. Use it when the
 * indices are valid by construction. See \ref UncompressOverview for a comparison
 * of the variants.
 *
 * \warning An index out of `[0, maskSize)` causes undefined behavior (an out-of-bounds
 * write). In debug builds, it is detected by an assertion.
 *
 * \tparam IndexVector Type of the input index vector.
 * \tparam MaskVector  Type of the returned mask vector (defaults to IndexVector).
 * \param indexVector  Vector of indices to mark as active (1).
 * \param maskSize     Size of the output mask. 0 means auto-detect from the input.
 * \param begin        Beginning of the range of \p indexVector to use.
 * \param end          End of the range of \p indexVector to use. 0 means the end of \p indexVector.
 * \return A new vector of size \p maskSize with 1 at each listed index.
 */
template<
   typename IndexVector,
   typename MaskVector = IndexVector,
   typename std::enable_if_t< IsArrayType< std::decay_t< IndexVector > >::value, bool > = true >
MaskVector
uncompressUnchecked(
   const IndexVector& indexVector,
   typename IndexVector::IndexType maskSize = 0,
   typename IndexVector::IndexType begin = 0,
   typename IndexVector::IndexType end = 0 )
{
   MaskVector maskVector;
   detail::uncompress_impl< false >( indexVector, maskVector, maskSize, begin, end );
   return maskVector;
}

/**
 * \brief Converts a list of indices into a boolean mask without checking the indices,
 *        filling an existing vector.
 *
 * Same as the filling overload of \ref TNL::Algorithms::uncompress, but the indices are
 * not checked to be in `[0, maskSize)`. See the returning overload of
 * \ref TNL::Algorithms::uncompressUnchecked for details.
 *
 * \warning An index out of `[0, maskSize)` causes undefined behavior (an out-of-bounds
 * write). In debug builds, it is detected by an assertion.
 *
 * \tparam IndexVector Type of the input index vector.
 * \tparam MaskVector  Type of the output mask vector.
 * \param indexVector  Vector of indices to mark as active (1).
 * \param maskVector   Output vector to fill.
 * \param maskSize     Size of the output mask. 0 means auto-detect from the input.
 * \param begin        Beginning of the range of \p indexVector to use.
 * \param end          End of the range of \p indexVector to use. 0 means the end of \p indexVector.
 */
template<
   typename IndexVector,
   typename MaskVector,
   typename std::enable_if_t<
      IsArrayType< std::decay_t< IndexVector > >::value && IsArrayType< std::decay_t< MaskVector > >::value,
      bool > = true >
void
uncompressUnchecked(
   const IndexVector& indexVector,
   MaskVector& maskVector,
   typename IndexVector::IndexType maskSize = 0,
   typename IndexVector::IndexType begin = 0,
   typename IndexVector::IndexType end = 0 )
{
   detail::uncompress_impl< false >( indexVector, maskVector, maskSize, begin, end );
}

}  // namespace TNL::Algorithms

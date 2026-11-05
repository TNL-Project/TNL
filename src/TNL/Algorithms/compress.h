// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "detail/Compress.hpp"
#include <type_traits>

namespace TNL::Algorithms {

// clang-format off
/**
 * \page CompressOverview Overview of Compress Functions
 *
 * \tableofcontents
 *
 * This page gives an overview of the \ref TNL::Algorithms::compress and
 * \ref TNL::Algorithms::compressOverwritingMarks functions, the differences between them and
 * guidance for choosing the right one.
 *
 * \section CompressWhatIsCompress What is Compress?
 *
 * **Compress** (also called *stream compaction*) converts a vector of marks
 * (0/1 or `false`/`true`) into a dense list of the indices of the marks equal
 * to 1. For example:
 *
 * ```
 * index:   0  1  2  3  4  5  6  7  8  9 10
 * marks:   1  0  0  0  0  1  0  1  1  0  0
 * output:  0  5  7  8
 * ```
 *
 * All variants work in parallel on any device. They compute the inclusive prefix
 * sum of the marks and then write each marked index to the position given by the
 * prefix sum. The inverse operation, which converts a list of indices back to marks,
 * is \ref TNL::Algorithms::uncompress (see \ref UncompressOverview).
 *
 * \section CompressFunctions Overview of the Variants
 *
 * The variants differ in three ways:
 * - **where the marks come from**: a lambda function or a vector,
 * - **how the output is returned**: as a new vector, or written into a vector
 *   the caller provides (in that case the function returns the number of marks),
 * - **where the prefix sum is computed**: in a temporary vector allocated inside
 *   the function, in the input vector itself, or in a vector the caller provides.
 *
 * | Function                                                | Marks given by | Output          | Prefix sum stored in           | Input marks |
 * |---------------------------------------------------------|----------------|-----------------|--------------------------------|-------------|
 * | `compress< OutputVector >( begin, end, marksFunction )` | lambda         | returned vector | temporary vector (allocated)   | --          |
 * | `compress( begin, end, marksFunction, output )`         | lambda         | `output`, count | temporary vector (allocated)   | --          |
 * | `compress( marks, begin, end )`                         | vector         | returned vector | temporary copy of marks        | preserved   |
 * | `compress( marks, output, begin, end )`                 | vector         | `output`, count | temporary copy of marks        | preserved   |
 * | `compress( marks, output, aux, begin, end )`            | vector         | `output`, count | `aux` (provided by the caller) | preserved   |
 * | `compressOverwritingMarks( marks, begin, end )`         | vector         | returned vector | `marks` (in place)             | overwritten |
 * | `compressOverwritingMarks( marks, output, begin, end )` | vector         | `output`, count | `marks` (in place)             | overwritten |
 *
 * \section CompressCommonRules Common Rules
 *
 * - Only the marks in the range `[begin, end)` are processed. For the vector variants,
 *   `end == 0` (the default) means the end of the marks vector.
 * - The output contains the original indices, not indices relative to `begin`.
 *   For example, compressing the marks above in the range `[4, 11)` gives `5 7 8`.
 * - The marks must be exactly 0 or 1. Other nonzero values give wrong results.
 * - The variants that take `output` (and `aux`) enlarge these vectors only when
 *   they are too small, and never shrink them. Elements of `output` behind the
 *   returned count are left unchanged.
 * - The type used for the prefix sum (the type of the marks vector, of `aux`, or
 *   `OutputVector` for the lambda variants) must be able to hold the number of marks.
 *
 * \section CompressChoosing Choosing a Variant
 *
 * - The marks are not stored, or are cheap to compute from other data:
 *   use the **lambda variants**.
 * - The function is called once, or rarely: the variants that **return a new vector**
 *   are the simplest to use.
 * - The function is called repeatedly, e.g. in every iteration of an iterative
 *   algorithm: pass `output` so it is not reallocated in every call, and avoid the
 *   temporary vector:
 *   - use **`compressOverwritingMarks`** if the marks are not needed afterwards,
 *   - use **`compress( marks, output, aux )`** if the marks must be preserved;
 *     `aux` is allocated once and reused.
 *
 * On GPUs, allocating and freeing device memory is expensive and usually synchronizes
 * the device, so avoiding the temporary vector matters most there.
 */
// clang-format on

/**
 * \brief This function compresses marks given by the lambda function.
 *
 * For each index in the range `[begin, end)` the lambda function is called.
 * The lambda function returns a mark (0/1 or true/false) for each index.
 * The output vector contains indices of marks equal to 1 or true.
 *
 * Performance note: The function stores internally the marks in a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressOverwritingMarks if it is possible.
 *
 * \tparam OutputVector is type of the output vector.
 * \param begin is the beginning of the range.
 * \param end is the end of the range.
 * \param marksFunction is a lambda function that returns a mark for each index.
 * \return OutputVector containing indices of marks 1 or true.
 *
 * \par Example
 * \include Algorithms/compressExample-lambda.cpp
 * \par Output
 * \include compressExample-lambda.out
 *
 */
template< typename OutputVector, typename BeginIndex, typename EndIndex, typename MarksFunction >
OutputVector
compress( BeginIndex begin, EndIndex end, MarksFunction&& marksFunction )
{
   return detail::compress< OutputVector >( begin, end, std::forward< MarksFunction >( marksFunction ) );
}

/**
 * \brief This function compresses marks given by the lambda function.
 *
 * For each index in the range `[begin, end)` the lambda function is called.
 * The lambda function returns a mark (0/1 or true/false) for each index.
 * The indices of marks equal 1 or true are stored in the output vector.
 * Only if the size of the output vector is smaller than the number of marks equal to 1 or true,
 * the size of the output vector is increased. Otherwise no reallocation is performed.
 * The function returns the number of marks equal to 1 or true.
 *
 * Performance note: The function stores internally the marks in a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressOverwritingMarks if it is possible.
 *
 * \param begin is the beginning of the range.
 * \param end is the end of the range.
 * \param marksFunction is a lambda function that returns a mark for each index.
 * \param outputVector is the output vector.
 * \return OutputVector::IndexType is the number of marks equal to 1 or true.
 *
 * \par Example
 * \include Algorithms/compressExample-lambda.cpp
 * \par Output
 * \include compressExample-lambda.out
 */
template< typename BeginIndex, typename EndIndex, typename MarksFunction, typename OutputVector >
auto
compress( BeginIndex begin, EndIndex end, MarksFunction&& marksFunction, OutputVector& outputVector ) ->
   typename OutputVector::IndexType
{
   return detail::compress( begin, end, std::forward< MarksFunction >( marksFunction ), outputVector );
}

/**
 * \brief This function compress the input vector.
 *
 * The function return a vector containing indices of marks equal to 1 or true
 * in the input vector within the range on indices `[begin, end)`.
 *
 * Performance note: The function copies internally the marks into a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressOverwritingMarks if it is possible.
 *
 * \tparam MarksVector is the type of the input vector.
 * \tparam OutputVector is the type of the output vector.
 * \param marksVector is the input vector.
 * \param begin  is the beginning of the range.
 * \param end is the end of the range.
 * \return OutputVector is the output vector containing indices of marks equal to 1 or true.
 *
 * \par Example
 * \include Algorithms/compressExample-vector.cpp
 * \par Output
 * \include compressExample-vector.out
 */
template<
   typename MarksVector,
   typename OutputVector = MarksVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
OutputVector
compress( const MarksVector& marksVector, BeginIndex begin = 0, EndIndex end = 0 )
{
   MarksVector aux( marksVector );
   return detail::compressVector< MarksVector, OutputVector >( aux, begin, end );
}

/**
 * \brief This function compress the input vector.
 *
 * The function return a vector containing indices of marks equal to 1 or true
 * in the input vector within the range on indices `[begin, end)`.
 * Only if the size of the output vector is smaller than the number of marks equal to 1 or true,
 * the size of the output vector is increased. Otherwise no reallocation is performed.
 * The function returns the number of marks equal to 1 or true.
 *
 * Performance note: The function copies internally the marks into a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressOverwritingMarks if it is possible.
 *
 * \tparam MarksVector is the type of the input vector.
 * \tparam OutputVector is the type of the output vector.
 * \param marksVector is the input vector.
 * \param outputVector is the output vector.
 * \param begin  is the beginning of the range.
 * \param end is the end of the range.
 * \return OutputVector is the output vector containing indices of marks equal to 1 or true.
 *
 * \par Example
 * \include Algorithms/compressExample-vector.cpp
 * \par Output
 * \include compressExample-vector.out
 */
template<
   typename MarksVector,
   typename OutputVector = MarksVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
auto
compress( const MarksVector& marksVector, OutputVector& outputVector, BeginIndex begin = 0, EndIndex end = 0 ) ->
   typename OutputVector::IndexType
{
   MarksVector aux( marksVector );
   return detail::compressVector( aux, outputVector, begin, end );
}

/**
 * \brief This function compress the input vector using a user-provided auxiliary vector.
 *
 * The indices of marks equal to 1 or true in the input vector within the range of
 * indices `[begin, end)` are stored in the output vector. The input vector is not modified.
 * The inclusive prefix sum of the marks is computed into the auxiliary vector `auxVector`,
 * which avoids the temporary allocation performed by the other overloads of
 * \ref TNL::Algorithms::compress. This is useful when the function is called repeatedly,
 * e.g. in iterative algorithms.
 *
 * Only if the size of the output vector is smaller than the number of marks equal to 1 or true,
 * the size of the output vector is increased. Similarly, only if the size of the auxiliary vector
 * is smaller than `end - begin`, the size of the auxiliary vector is increased. Otherwise no
 * reallocation is performed. The content of the auxiliary vector is overwritten.
 * The function returns the number of marks equal to 1 or true.
 *
 * \tparam MarksVector is the type of the input vector.
 * \tparam OutputVector is the type of the output vector.
 * \tparam AuxVector is the type of the auxiliary vector. It must be allocated on the same
 *         device as the input vector and its value type must be able to hold the number
 *         of marks.
 * \param marksVector is the input vector.
 * \param outputVector is the output vector.
 * \param auxVector is the auxiliary vector for the prefix sum of the marks.
 * \param begin  is the beginning of the range.
 * \param end is the end of the range.
 * \return the number of marks equal to 1 or true.
 */
template<
   typename MarksVector,
   typename OutputVector,
   typename AuxVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && IsArrayType< OutputVector >::value && IsArrayType< AuxVector >::value
         && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
auto
compress(
   const MarksVector& marksVector,
   OutputVector& outputVector,
   AuxVector& auxVector,
   BeginIndex begin = 0,
   EndIndex end = 0 ) -> typename OutputVector::IndexType
{
   return detail::compressVector( marksVector, outputVector, auxVector, begin, end );
}

/**
 * \brief This function compress the input vector.
 *
 * The function return a vector containing indices of marks equal to 1 or true
 * in the input vector within the range on indices `[begin, end)`.
 *
 * Warning: The function uses the input vector `marksVector` for internal computations.
 * The content of the input vector is therefore modified. On the other hand, the
 * performance of this function is better compared to the function \ref TNL::Algorithms::compress.
 *
 * \tparam MarksVector is the type of the input vector.
 * \tparam OutputVector is the type of the output vector.
 * \param marksVector is the input vector.
 * \param begin  is the beginning of the range.
 * \param end is the end of the range.
 * \return OutputVector is the output vector containing indices of marks equal to 1 or true.
 *
 * \par Example
 * \include Algorithms/compressExample-vector.cpp
 * \par Output
 * \include compressExample-vector.out
 */
template<
   typename MarksVector,
   typename OutputVector = MarksVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
OutputVector
compressOverwritingMarks( MarksVector& marksVector, BeginIndex begin = 0, EndIndex end = 0 )
{
   return detail::compressVector< MarksVector, OutputVector >( marksVector, begin, end );
}

/**
 * \brief This function compress the input vector.
 *
 * The function return a vector containing indices of marks equal to 1 or true
 * in the input vector within the range on indices `[begin, end)`.
 * Only if the size of the output vector is smaller than the number of marks equal to 1 or true,
 * the size of the output vector is increased. Otherwise no reallocation is performed.
 * The function returns the number of marks equal to 1 or true.
 *
 * Warning: The function uses the input vector `marksVector` for internal computations.
 * The content of the input vector is therefore modified. On the other hand, the
 * performance of this function is better compared to the function \ref TNL::Algorithms::compress.
 *
 * \tparam MarksVector is the type of the input vector.
 * \tparam OutputVector is the type of the output vector.
 * \param marksVector is the input vector.
 * \param outputVector is the output vector.
 * \param begin  is the beginning of the range.
 * \param end is the end of the range.
 * \return OutputVector is the output vector containing indices of marks equal to 1 or true.
 *
 * \par Example
 * \include Algorithms/compressExample-vector.cpp
 * \par Output
 * \include compressExample-vector.out
 */
template<
   typename MarksVector,
   typename OutputVector = MarksVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
auto
compressOverwritingMarks( MarksVector& marksVector, OutputVector& outputVector, BeginIndex begin = 0, EndIndex end = 0 ) ->
   typename OutputVector::IndexType
{
   return detail::compressVector( marksVector, outputVector, begin, end );
}

/**
 * \brief Deprecated alias of \ref TNL::Algorithms::compressOverwritingMarks.
 *
 * \deprecated Use \ref TNL::Algorithms::compressOverwritingMarks instead.
 */
template<
   typename MarksVector,
   typename OutputVector = MarksVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
[[deprecated( "Use TNL::Algorithms::compressOverwritingMarks instead" )]] OutputVector
compressFast( MarksVector& marksVector, BeginIndex begin = 0, EndIndex end = 0 )
{
   return compressOverwritingMarks< MarksVector, OutputVector >( marksVector, begin, end );
}

/**
 * \brief Deprecated alias of \ref TNL::Algorithms::compressOverwritingMarks.
 *
 * \deprecated Use \ref TNL::Algorithms::compressOverwritingMarks instead.
 */
template<
   typename MarksVector,
   typename OutputVector = MarksVector,
   typename BeginIndex = typename OutputVector::IndexType,
   typename EndIndex = BeginIndex,
   typename std::enable_if_t<
      IsArrayType< MarksVector >::value && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex >,
      bool > = true >
[[deprecated( "Use TNL::Algorithms::compressOverwritingMarks instead" )]] auto
compressFast( MarksVector& marksVector, OutputVector& outputVector, BeginIndex begin = 0, EndIndex end = 0 ) ->
   typename OutputVector::IndexType
{
   return compressOverwritingMarks( marksVector, outputVector, begin, end );
}

}  // namespace TNL::Algorithms

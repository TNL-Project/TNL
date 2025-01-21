// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "detail/Compress.hpp"
#include <type_traits>

namespace TNL::Algorithms {

/**
 * \brief This function compresses marks given by the lambda function.
 *
 * For each index in the range `[begin, end)` the lambda function is called.
 * The lambda function returns a mark (0/1 or true/false) for each index.
 * The output vector contains indices of marks equal to 1 or true.
 *
 * Performance note: The function stores internally the marks in a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressFast if it is possible.
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
 * this overhead, use \ref TNL::Algorithms::compressFast if it is possible.
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
 * in the input vector within the range on indices [\e begin, \e end).
 *
 * Performance note: The function copies internally the marks into a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressFast if it is possible.
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
template< typename MarksVector,
          typename OutputVector = MarksVector,
          typename BeginIndex = typename OutputVector::IndexType,
          typename EndIndex = BeginIndex,
          typename T = std::enable_if_t< IsArrayType< MarksVector >::value
                                         && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex > > >
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
 * in the input vector within the range on indices [\e begin, \e end).
 * Only if the size of the output vector is smaller than the number of marks equal to 1 or true,
 * the size of the output vector is increased. Otherwise no reallocation is performed.
 * The function returns the number of marks equal to 1 or true.
 *
 * Performance note: The function copies internally the marks into a temporary vector
 * which requires dynamic memory allocation and deallocation. If you want to avoid
 * this overhead, use \ref TNL::Algorithms::compressFast if it is possible.
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
template< typename MarksVector,
          typename OutputVector = MarksVector,
          typename BeginIndex = typename OutputVector::IndexType,
          typename EndIndex = BeginIndex,
          typename T = std::enable_if_t< IsArrayType< MarksVector >::value > >
auto
compress( const MarksVector& marksVector, OutputVector& outputVector, BeginIndex begin = 0, EndIndex end = 0 ) ->
   typename OutputVector::IndexType
{
   MarksVector aux( marksVector );
   return detail::compressVector( aux, outputVector, begin, end );
}

/**
 * \brief This function compress the input vector.
 *
 * The function return a vector containing indices of marks equal to 1 or true
 * in the input vector within the range on indices [\e begin, \e end).
 *
 * Warning: The function uses the input vector \e marksVector for internal computations.
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
template< typename MarksVector,
          typename OutputVector = MarksVector,
          typename BeginIndex = typename OutputVector::IndexType,
          typename EndIndex = BeginIndex,
          typename T = std::enable_if_t< IsArrayType< MarksVector >::value
                                         && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex > > >
OutputVector
compressFast( MarksVector& marksVector, BeginIndex begin = 0, EndIndex end = 0 )
{
   return detail::compressVector< MarksVector, OutputVector >( marksVector, begin, end );
}

/**
 * \brief This function compress the input vector.
 *
 * The function return a vector containing indices of marks equal to 1 or true
 * in the input vector within the range on indices [\e begin, \e end).
 * Only if the size of the output vector is smaller than the number of marks equal to 1 or true,
 * the size of the output vector is increased. Otherwise no reallocation is performed.
 * The function returns the number of marks equal to 1 or true.
 *
 * Warning: The function uses the input vector \e marksVector for internal computations.
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
template< typename MarksVector,
          typename OutputVector = MarksVector,
          typename BeginIndex = typename OutputVector::IndexType,
          typename EndIndex = BeginIndex,
          typename T = std::enable_if_t< IsArrayType< MarksVector >::value
                                         && std::is_integral_v< BeginIndex > && std::is_integral_v< EndIndex > > >
auto
compressFast( MarksVector& marksVector, OutputVector& outputVector, BeginIndex begin = 0, EndIndex end = 0 ) ->
   typename OutputVector::IndexType
{
   return detail::compressVector( marksVector, outputVector, begin, end );
}

}  // namespace TNL::Algorithms

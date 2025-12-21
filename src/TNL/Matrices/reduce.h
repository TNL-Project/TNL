// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Matrices {

// TODO: The following is incomplete implementation of traversing functions for matrices. It is necessary
// for traversing of graphs and it currently supports sparse and dense matrices only. It should be extended to support
// other matrix types as well. Also, documentation and examples should be added.

/**
 * \page MatrixReductionLambdas Matrix Reduction Lambda Function Reference
 *
 * This page provides a comprehensive reference for all lambda function signatures used
 * in matrix reduction operations.
 *
 * \tableofcontents
 *
 * \section FetchLambdas Fetch Lambda Functions
 *
 * The \e fetch lambda is used to extract and transform values from matrix elements during reduction.
 *
 * \subsection FetchLambda_NonConst For Non-Const Matrices
 *
 * For **non-const matrices sparse matrices** (\ref TNL::Matrices::SparseMatrix and \ref TNL::Matrices::SparseMatrixView)
 * the signature of the \e fetch lambda is:
 *
 * ```cpp
 * auto fetch = [] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& value ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e rowIdx - The index of the matrix row
 * - \e columnIdx - The index of the matrix column (can be modified)
 * - \e value - The value of the matrix element (can be modified)
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * For other types matrices like dense (\ref TNL::Matrices::DenseMatrix, \ref TNL::Matrices::DenseMatrixView),
 * tridiagonal matrices (\ref TNL::Matrices::TridiagonalMatrix, \ref TNL::Matrices::TridiagonalMatrixView) or
 * multidiagonal (\ref TNL::Matrices::MultidiagonalMatrix, \ref TNL::Matrices::MultidiagonalMatrixView),
 * the column index is defined implicitly and cannot be changed even for non-constant matrices. The signature
 * then reads as:
 *
 * ```cpp
 * auto fetch = [] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, RealType& value ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e rowIdx - The index of the matrix row
 * - \e columnIdx - The index of the matrix column (passed by value)
 * - \e value - The value of the matrix element (can be modified)
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * \subsection FetchLambda_Const For Const Matrices
 *
 * ```cpp
 * auto fetch = [] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e rowIdx - The index of the matrix row
 * - \e columnIdx - The index of the matrix column (passed by value)
 * - \e value - The value of the matrix element (const reference)
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * \section ReductionLambdas Reduction Lambda Functions
 *
 * The \e reduction lambda defines how values are combined during the reduction operation.
 *
 * \subsection ReductionLambda_Basic Basic Reduction (Without Arguments)
 *
 * ```cpp
 * auto reduction = [=] __cuda_callable__ ( const FetchValue& a, const FetchValue& b ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e a - First value to be reduced
 * - \e b - Second value to be reduced
 * - Returns: The result of reducing \e a and \e b
 *
 * \subsection ReductionLambda_WithArgument Reduction With Argument (Position Tracking)
 *
 * ```cpp
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) -> Result { ... }
 * ```
 *
 * **Parameters:**
 * - \e a - First value to be reduced (mutable reference)
 * - \e b - Second value to be reduced (const reference)
 * - \e aIdx - Index/position associated with value \e a (mutable reference for tracking)
 * - \e bIdx - Index/position associated with value \e b (const reference)
 *
 * Note: This variant is used when you need to track which element produced the final result
 * (e.g., finding the maximum value and its position).
 *
 * \section KeepLambdas Keep Lambda Functions
 *
 * The \e keep lambda is used to store the final reduction result for each row.
 *
 * \subsection KeepLambda_Basic Basic Keep (Row Index Only)
 *
 * ```cpp
 * auto keep = [=] __cuda_callable__ ( IndexType rowIdx, const FetchValue& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e rowIdx - The index of the row
 * - \e value - The result of the reduction for this row
 *
 * \subsection KeepLambda_WithLocalIdx Keep With Argument (Position Tracking)
 *
 * ```cpp
 * auto keep = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const Value& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e rowIdx - The index of the row
 * - \e localIdx - The local index of the element within the row (when tracking positions)
 * - \e columnIdx - The column index of the element within the row (when tracking positions)
 * - \e value - The result of the reduction for this row
 *
 * \subsection KeepLambda_WithIndexArray Keep With Row Index Array Or Condition
 *
 * ```cpp
 * auto keep = [=] __cuda_callable__ ( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e indexOfRowIdx - The position within the \e rowIndexes array or the rank in the set of rows for which the condition was
 * true.
 * - \e rowIdx - The actual index of the row
 * - \e value - The result of the reduction for this row
 *
 * \subsection KeepLambda_WithIndexArrayAndLocalIdx Keep With Row Index Array and Local Index
 *
 * ```cpp
 * auto keep = [=] __cuda_callable__ ( IndexType indexOfRowIdx, IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const
 * Value& value ) { ...
 * }
 * ```
 *
 * **Parameters:**
 * - \e indexOfRowIdx - The position within the \e rowIndexes array or the rank in the set of rows for which the condition was
 * true.
 * - \e rowIdx - The actual index of the row
 * - \e localIdx - The position of the element within the row
 * - \e columnIdx - The column index of the element within the row
 * - \e value - The result of the reduction for this row
 *
 * \section ConditionLambdas Condition Lambda Functions
 *
 * The \e condition lambda determines which rows should be processed (used in "If" variants).
 *
 * \subsection ConditionLambda Condition Check
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType rowIdx ) -> bool { ... }
 * ```
 *
 * **Parameters:**
 * - \e rowIdx - The index of the matrix row
 * - Returns: \e true if the row should be processed, \e false otherwise
 *
 * \section ReductionFunctionObjects Reduction Function Objects
 *
 * Instead of lambda functions, reduction operations can also be specified using function objects
 * from \ref TNL::Algorithms::Segments::ReductionFunctionObjects or
 * \ref TNL::Algorithms::Segments::ReductionFunctionObjectsWithArgument.
 *
 * When using function objects:
 * - They must provide a static template method \e getIdentity to automatically deduce the identity value
 * - For WithArgument variants, they must be instances of \ref ReductionFunctionObjectsWithArgument
 * - Common examples: \e Min, \e Max, \e Sum, \e Product, \e MinWithArg, \e MaxWithArg
 */

/**
 * \brief Performs parallel reduction within each matrix row over all rows.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRows( Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRows( const Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows
 * with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forAllElements.cpp
 * \par Output
 * \include MatrixExample_forAllElements.out
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRows( Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows
 * with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forAllElements.cpp
 * \par Output
 * \include MatrixExample_forAllElements.out
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRows( const Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for the reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRows( Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for the reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRows( const Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes
 * with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRows( Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes
 * with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRows( const Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of row indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of row indexes where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed for reduction.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithIndexArray.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRows( Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithIndexArray.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRows( const Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes
 * with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithIndexArray.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forRowsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forRowsWithIndexes.out
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRows( Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes
 * with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithIndexArray.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forRowsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forRowsWithIndexes.out
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRows( const Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forAllRowsIf.cpp
 * \par Output
 * \include MatrixExample_forAllRowsIf.out
 */
template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsIf( Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forAllRowsIf.cpp
 * \par Output
 * \include MatrixExample_forAllRowsIf.out
 */
template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsIf( const Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition
 * with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forAllRowsIf.cpp
 * \par Output
 * \include MatrixExample_forAllRowsIf.out
 */
template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIf( Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition
 * with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forAllRowsIf.cpp
 * \par Output
 * \include MatrixExample_forAllRowsIf.out
 */
template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIf( const Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue >
void
reduceRowsIf( Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              const FetchValue& identity,
              Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition (const
 * version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue >
void
reduceRowsIf( const Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              const FetchValue& identity,
              Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition
 * with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIf( Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition
 * with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIf( const Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows while
 *  returning also the position of the element of interest.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsWithArgument.out
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsWithArgument(
   Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows while
 *  returning also the position of the element of interest (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsWithArgument.out
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsWithArgument(
   const Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows while
 *  returning also the position of the element of interest with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsWithArgument.out
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsWithArgument(
   Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows while
 *  returning also the position of the element of interest with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsWithArgument.out
 */
template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsWithArgument(
   const Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation with argument. See \ref ReductionLambda_WithArgument.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRowsWithArgument( Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes while
 *  returning also the position of the element of interest (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation with argument. See \ref ReductionLambda_WithArgument.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRowsWithArgument( const Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes while
 *  returning also the position of the element of interest with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation with argument. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRowsWithArgument( Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes while
 *  returning also the position of the element of interest with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation with argument. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceRowsWithArgument( const Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithIndexArrayAndLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsWithArgument.out
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRowsWithArgument( Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes while
 *  returning also the position of the element of interest (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithIndexArrayAndLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsWithArgument.out
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRowsWithArgument( const Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes while
 *  returning also the position of the element of interest with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithIndexArrayAndLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsWithArgument.out
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRowsWithArgument( Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within matrix rows specified by a given set of row indexes while
 *  returning also the position of the element of interest with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the rows to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param rowIndexes The array containing the indexes of the rows to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithIndexArrayAndLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsWithArgument.out
 */
template< typename Matrix,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceRowsWithArgument( const Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition while
 *  returning also the position of the element of interest.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param identity The identity element for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsIfWithArgument.out
 */
template< typename Matrix,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Matrix::RealType >() ) ) >
void
reduceAllRowsIfWithArgument(
   Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition while
 *  returning also the position of the element of interest (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param identity The identity element for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsIfWithArgument.out
 */
template< typename Matrix,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Matrix::RealType >() ) ) >
void
reduceAllRowsIfWithArgument(
   const Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition while
 *  returning also the position of the element of interest with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsIfWithArgument.out
 */
template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIfWithArgument(
   Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over all rows based on a condition while
 *  returning also the position of the element of interest with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceAllRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceAllRowsIfWithArgument.out
 */
template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIfWithArgument(
   const Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition while
 *  returning also the position of the element of interest.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param identity The identity element for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsIfWithArgument.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Matrix::RealType >() ) ) >
void
reduceRowsIfWithArgument(
   Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition while
 *  returning also the position of the element of interest (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param identity The identity element for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsIfWithArgument.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Matrix::RealType >() ) ) >
void
reduceRowsIfWithArgument(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition while
 *  returning also the position of the element of interest with automatic identity deduction.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsIfWithArgument.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIfWithArgument(
   Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each matrix row over a given range of row indexes based on a condition while
 *  returning also the position of the element of interest with automatic identity deduction (const version).
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Keep The type of the lambda function used for storing results from individual rows.
 *
 * \param matrix The matrix on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of rows where the reduction will be performed.
 * \param condition Lambda function for row condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param keep Lambda function for storing results with position tracking. See \ref KeepLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Reduce/MatrixExample_reduceRowsIfWithArgument.cpp
 * \par Output
 * \include MatrixExample_reduceRowsIfWithArgument.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIfWithArgument(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Keep&& keep,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Matrices

#include "reduce.hpp"

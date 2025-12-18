// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

// TODO: The following is incomplete implementation of traversing functions for matrices. It is necessary
// for traversing of graphs and it currently supports sparse and dense matrices only. It should be extended to support
// other matrix types as well. Also, documentation and examples should be added.

/**
 * \page MatrixTraversalLambdas Matrix Traversal Lambda Functions Reference
 *
 * This page documents the lambda function signatures used in matrix traversal operations.
 *
 * \section TraversalFunction_NonConst Traversal Function (Non-Const Matrix)
 *
 * For **non-const matrices sparse matrices** (\ref TNL::Matrices::SparseMatrix and \ref TNL::Matrices::SparseMatrixView),
 * the traversal function has full access to modify element indices and values:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType& columnIdx, RealType& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - `rowIdx` - The index of the matrix row to which the element belongs
 * - `localIdx` - The rank/position of the element within the matrix row (0-based)
 * - `columnIdx` - Reference to the column index (can be modified)
 * - `value` - Reference to the element value (can be modified)
 *
 * For other types matrices like dense (\ref TNL::Matrices::DenseMatrix, \ref TNL::Matrices::DenseMatrixView),
 * tridiagonal matrices (\ref TNL::Matrices::TridiagonalMatrix, \ref TNL::Matrices::TridiagonalMatrixView) or
 * multidiagonal (\ref TNL::Matrices::MultidiagonalMatrix, \ref TNL::Matrices::MultidiagonalMatrixView),
 * the column index is defined implicitly and cannot be changed even for non-constant matrices. The signature
 * then reads as:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, RealType& value ) { ... }
 *
 * ```
 * **Parameters:**
 * - `rowIdx` - The index of the matrix row to which the element belongs
 * - `localIdx` - The rank/position of the element within the matrix row (0-based)
 * - `columnIdx` - Column index
 * - `value` - Reference to the element value (can be modified)
 *
 * \section TraversalFunction_Const Traversal Function (Const Matrix)
 *
 * For const matrices, the traversal function has read-only access:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) {
 * ... }
 * ```
 *
 * **Parameters:**
 * - `rowIdx` - The index of the matrix row to which the element belongs
 * - `localIdx` - The rank/position of the element within the matrix row (0-based)
 * - `columnIdx` - The column index (read-only)
 * - `value` - Const reference to the element value (read-only)
 *
 * \section TraversalRowFunction_NonConst Row Traversal Function (Non-Const Matrix)
 *
 * When traversing entire rows with row-level operations, the function receives the row index:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( typename Matrix::RowView row ) { ... }
 * ```
 *
 * **Parameters:**
 * - `rowIdx` - The index of the matrix row being processed
 *
 * \section TraversalRowFunction_Const Row Traversal Function (Const Matrix)
 *
 * Same signature for const matrices:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( typename Matrix::ConstRowView row ) { ... }
 * ```
 *
 * **Parameters:**
 * - `rowIdx` - The index of the matrix row being processed
 *
 * \section TraversalConditionLambda Condition Lambda
 *
 * For conditional traversal operations (`forElementsIf`, `forRowsIf`), a condition function determines
 * which rows to process:
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType rowIdx ) -> bool { ... }
 * ```
 *
 * **Parameters:**
 * - `rowIdx` - The index of the matrix row to check
 *
 * **Returns:**
 * - `true` if the row should be processed, `false` to skip it
 *
 */

namespace TNL::Matrices {

/**
 * \brief Iterates in parallel over all elements in the given range of matrix rows and
 * applies the specified lambda function.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows whose elements we want to process using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows whose elements we want to process using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrix whose elements will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( Matrix& matrix,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of **constant matrix** in the given range of matrix rows and
 * applies the specified lambda function.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows whose elements we want to process using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows whose elements we want to process using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrix whose elements will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Matrix& matrix,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of **all** matrix rows and
 * applies the specified lambda function.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix, typename Function >
void
forAllElements( Matrix& matrix,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of **all** matrix rows of **constant matrix** and
 * applies the specified lambda function.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix, typename Function >
void
forAllElements( const Matrix& matrix,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of matrix rows with the given indexes and
 * applies the specified lambda function.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of row indexes whose elements will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of row indexes whose elements will be processed using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes whose elements
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forElementsWithIndexes.out
 */
template< typename Matrix, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( Matrix& matrix,
             const Array& rowIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of matrix rows with the given indexes and
 * applies the specified lambda function. This function is for **constant matrices**.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of row indexes whose elements will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of row indexes whose elements will be processed using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes whose elements
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forElementsWithIndexes.out
 */
template< typename Matrix, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Matrix& matrix,
             const Array& rowIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of matrix rows with the given indexes and
 * applies the specified lambda function.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forElementsWithIndexes.out
 */
template< typename Matrix, typename Array, typename Function >
void
forElements( Matrix& matrix,
             const Array& rowIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of matrix rows with the given indexes and
 * applies the specified lambda function. This function is for **constant matrices**.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forElementsWithIndexes.out
 */
template< typename Matrix, typename Array, typename Function >
void
forElements( const Matrix& matrix,
             const Array& rowIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements in a given range of rows based on a condition.
 *
 * For each matrix row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, all elements of the row are traversed,
 * and the specified lambda function is applied to each element. If the condition lambda function returns
 * \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows whose elements will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows whose elements will be processed using the lambda function.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param condition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsIf.cpp
 * \par Output
 * \include MatrixExample_forElementsIf.out
 */
template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( Matrix& matrix,
               IndexBegin begin,
               IndexEnd end,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements in a given range of rows based on a condition.
 * This function is for **constant matrices**.
 *
 * For each matrix row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, all elements of the row are traversed,
 * and the specified lambda function is applied to each element. If the condition lambda function returns
 * \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows whose elements will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows whose elements will be processed using the lambda function.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of rows whose elements
 *    will be processed using the lambda function.
 * \param condition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsIf.cpp
 * \par Output
 * \include MatrixExample_forElementsIf.out
 */
template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Matrix& matrix,
               IndexBegin begin,
               IndexEnd end,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of **all** matrix rows based on a condition.
 *
 * For each matrix row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, all elements of the row are traversed,
 * and the specified lambda function is applied to each element. If the condition lambda function returns
 * \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param condition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsIf.cpp
 * \par Output
 * \include MatrixExample_forElementsIf.out
 */
template< typename Matrix, typename Condition, typename Function >
void
forAllElementsIf( Matrix& matrix,
                  Condition&& condition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of **all** matrix rows based on a condition.
 * This function is for **constant matrices**.
 *
 * For each matrix row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, all elements of the row are traversed,
 * and the specified lambda function is applied to each element. If the condition lambda function returns
 * \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param matrix The matrixwhose elements will be processed using the lambda function.
 * \param condition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each element. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forElementsIf.cpp
 * \par Output
 * \include MatrixExample_forElementsIf.out
 */
template< typename Matrix, typename Condition, typename Function >
void
forAllElementsIf( const Matrix& matrix,
                  Condition&& condition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over matrix rows within the specified range of row indexes
 * and applies the given lambda function to each row.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of matrix rows on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of matrix rows on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of matrix rows
 *    that will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of matrix rows
 *    that will be processed using the lambda function.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/MatrixExample_forRows-2.cpp
 * \par Output
 * \include MatrixExample_forRows-2.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forRows( Matrix& matrix,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over matrix rows within the specified range of row indexes
 * and applies the given lambda function to each row. This function is for **constant matrices**.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of matrix rows on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of matrix rows on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of matrix rows
 *    that will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of matrix rows
 *    that will be processed using the lambda function.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/MatrixExample_forRows-2.cpp
 * \par Output
 * \include MatrixExample_forRows-2.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forRows( const Matrix& matrix,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** matrix rows and applies the given lambda function to each row.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/MatrixExample_forRows-2.cpp
 * \par Output
 * \include MatrixExample_forRows-2.out
 */
template< typename Matrix, typename Function >
void
forAllRows( Matrix& matrix,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** matrix rows and applies the given lambda function to each row.
 * This function is for **constant matrices**.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/MatrixExample_forRows-2.cpp
 * \par Output
 * \include MatrixExample_forRows-2.out
 */
template< typename Matrix, typename Function >
void
forAllRows( const Matrix& matrix,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over matrix rows with the given indexes and applies the specified
 * lambda function to each row.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forRowsWithIndexes.out
 */
template< typename Matrix,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forRows( Matrix& matrix,
         const Array& rowIndexes,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over matrix rows with the given indexes and applies the specified
 * lambda function to each row. This function is for **constant matrices**.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forRowsWithIndexes.out
 */
template< typename Matrix,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forRows( const Matrix& matrix,
         const Array& rowIndexes,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over matrix rows with the given indexes and applies the specified
 * lambda function to each row.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forRowsWithIndexes.out
 */
template< typename Matrix, typename Array, typename Function, typename T = std::enable_if_t< IsArrayType< Array >::value > >
void
forRows( Matrix& matrix,
         const Array& rowIndexes,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over matrix rows with the given indexes and applies the specified
 * lambda function to each row. This function is for **constant matrices**.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam Array The type of the array containing the indexes of the matrix rows to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param rowIndexes The array containing the indexes of the matrix rows to iterate over.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsWithIndexes.cpp
 * \par Output
 * \include MatrixExample_forRowsWithIndexes.out
 */
template< typename Matrix, typename Array, typename Function, typename T = std::enable_if_t< IsArrayType< Array >::value > >
void
forRows( const Matrix& matrix,
         const Array& rowIndexes,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );
/**
 * \brief Iterates in parallel over rows within the given range of row indexes, applying a condition
 * to determine whether each row should be processed.
 *
 * For each row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the row.
 * If the condition lambda function returns \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam RowCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param rowCondition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename RowCondition,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forRowsIf( Matrix& matrix,
           IndexBegin begin,
           IndexEnd end,
           RowCondition&& rowCondition,
           Function&& function,
           Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over rows within the given range of row indexes, applying a condition
 * to determine whether each row should be processed. This function is for **constant matrices**.
 *
 * For each row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the row.
 * If the condition lambda function returns \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of rows on which the lambda function will be applied.
 * \tparam RowCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param rowCondition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename RowCondition,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forRowsIf( const Matrix& matrix,
           IndexBegin begin,
           IndexEnd end,
           RowCondition&& rowCondition,
           Function&& function,
           Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** matrix rows, applying a condition
 * to determine whether each row should be processed.
 *
 * For each row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the row.
 * If the condition lambda function returns \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam RowCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param rowCondition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix, typename RowCondition, typename Function >
void
forAllRowsIf( Matrix& matrix,
              RowCondition&& rowCondition,
              Function&& function,
              Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** matrix rows, applying a condition
 * to determine whether each row should be processed. This function is for **constant matrices**.
 *
 * For each row, a condition lambda function is evaluated based on the row index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the row.
 * If the condition lambda function returns \e false, the row is skipped.
 *
 * \tparam Matrix The type of the matrix.
 * \tparam RowCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each row.
 *
 * \param matrix The matrixon which the lambda function will be applied.
 * \param rowCondition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/MatrixExample_forRowsIf.cpp
 * \par Output
 * \include MatrixExample_forRowsIf.out
 */
template< typename Matrix, typename RowCondition, typename Function >
void
forAllRowsIf( const Matrix& matrix,
              RowCondition&& rowCondition,
              Function&& function,
              Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Matrices

#include "traverse.hpp"

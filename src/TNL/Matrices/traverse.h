// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Matrices {

// TODO: The following is incomplete implementation of traversing functions for matrices. It is necessary
// for traversing of graphs and it currently supports sparse and dense matrices only. It should be extended to support
// other matrix types as well. Also, documentation and examples should be added.

/**
 * \page MatrixTraversalOverview Overview of Matrix Traversal Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all traversal functions available for matrix operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section MatrixTraversalWhatIsTraversal What is Matrix Traversal?
 *
 * **Matrix traversal** operations apply a user-defined function to matrix elements or rows.
 * Unlike reductions (which compute results), traversals are used for side effects:
 * - Modifying matrix elements in-place
 * - Performing computations that depend on element positions
 * - Implementing custom matrix algorithms
 *
 * \section MatrixTraversalFunctionCategories Function Categories
 *
 * Matrix traversal functions are organized along three main dimensions:
 *
 * \subsection MatrixTraversalConstVsNonConst Const vs. Non-Const Matrix
 *
 * | Category | Matrix Modifiable? | Use Case |
 * |----------|-------------------|----------|
 * | **Non-const** | Yes | Can modify matrix elements and structure |
 * | **Const** | No | Read-only access to matrix elements |
 *
 * Note: Each traversal function has **both const and non-const overloads**.
 *
 * \subsection MatrixTraversalElementVsRow Element-wise vs. Row-wise Traversal
 *
 * | Category | Operates On | Lambda Parameter | Use Case |
 * |----------|------------|------------------|----------|
 * | **Element-wise** (`forElements`, `forAllElements`) | Individual elements | Element indices & values | Operate on each
 * element separately |
 * | **Row-wise** (`forRows`, `forAllRows`) | Whole rows | RowView object | Operate on rows as units |
 *
 * \subsection MatrixTraversalScopeAndConditional Scope and Conditional Variants
 *
 * Similar to other matrix operations, traversal functions have different scope and conditional variants.
 *
 * | Scope | Rows Processed | Parameters |
 * |-------|---------------|------------|
 * | **All** | All rows | No range/array parameters |
 * | **Range** | Rows [begin, end) | `begin` and `end` indices |
 * | **Array** | Specific rows | Array of row indices |
 * | **If** | Rows filtered by a condition | Process rows based on row-level properties |
 *
 * \section MatrixTraversalElementFunctions Element-wise Traversal Functions
 *
 * These functions iterate over individual elements within matrix rows:
 *
 * \subsection MatrixTraversalBasicElementFunctions Basic Element Traversal
 *
 * | Function | Rows Processed | Description | Overloads |
 * |----------|---------------|-------------|----------|
 * | \ref forAllElements | All rows | Process all elements in all rows | const & non-const |
 * | \ref forElements (range) | Rows [begin, end) | Process elements in row range | const & non-const |
 * | \ref forElements (array) | Rows in array | Process elements in specified rows | const & non-const |
 * | \ref forAllElementsIf | All rows | Row-level condition | const & non-const |
 * | \ref forElementsIf | Rows [begin, end) | Row-level condition | const & non-const |
 *
 * **When to use:**
 * - Matrix elements assembly and updates
 * - Element-level operations and transformations
 *
 * \section MatrixTraversalRowFunctions Row-wise Traversal Functions
 *
 * These functions iterate over rows as whole units using \e RowView:
 *
 * \subsection MatrixTraversalBasicRowFunctions Basic Row Traversal
 *
 * | Function | Rows Processed | Description | Overloads |
 * |----------|---------------|-------------|----------|
 * | \ref forAllRows | All rows | Process all rows | const & non-const |
 * | \ref forRows (range) | Rows [begin, end) | Process rows in range | const & non-const |
 * | \ref forRows (array) | Rows in array | Process specified rows | const & non-const |
 * | \ref forAllRowsIf | All rows | Row-level condition | const & non-const |
 * | \ref forRowsIf | Rows [begin, end) | Row-level condition | const & non-const |
 *
 * **When to use:**
 * - Row-level operations (scaling, normalization)
 *
 * \section MatrixTraversalParameters Common Parameters
 *
 * All traversal functions share these common parameters:
 *
 * - **matrix**: The matrix to traverse (const or non-const)
 * - **function**: Lambda function to apply (see \ref TraversalFunction_NonConst, \ref TraversalFunction_Const, \ref
 * TraversalRowFunction_NonConst, or \ref TraversalRowFunction_Const)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Additional parameters:
 * - **Scope variants**: `begin`, `end` (range) or `rowIndexes` (array)
 * - **If variants**: `condition` lambda for filtering (see \ref TraversalConditionLambda)
 *
 * \section MatrixTraversalUsageGuidelines Usage Guidelines
 *
 * **Matrix type considerations:**
 *
 * - **Sparse matrices** (SparseMatrix, SparseMatrixView):
 *   - Element traversal: Can modify even column indices for non-const matrices
 *   - Row traversal: Use RowView for efficient sparse operations
 *
 * - **Dense matrices** (DenseMatrix, DenseMatrixView):
 *   - Column indices are implicit
 *   - Element traversal processes all elements in each row
 *
 * - **Structured matrices** (Tridiagonal, Multidiagonal) (NOT IMPLEMENTED YET):
 *   - Column indices follow fixed patterns
 *   - Element traversal only processes non-zero structure
 *
 * **Performance considerations:**
 * - Element-wise traversal is is parallel within rows, i.e. one matrix row can be processed by multiple threads
 * - Row-wise traversal never maps multiple threads to the same row. Row-wise traversal is therefore preferred when
 *   row-level context is needed.
 *
 * \section MatrixTraversalRelatedPages Related Pages
 *
 * - \ref MatrixTraversalLambdas - Detailed lambda function signatures
 * - \ref MatrixReductionOverview - Matrix reduction operations
 */

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
 * \include Matrices/Traverse/MatrixExample_forElements.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElements.cpp
 * \par Output
 * \include MatrixExample_forElements.out
 */
template< typename Matrix, typename Function >
void
forAllElements( const Matrix& matrix,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

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
 * \include Matrices/Traverse/MatrixExample_forElements.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElements.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsIf.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsIf.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsIf.cpp
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
 * \include Matrices/Traverse/MatrixExample_forElementsIf.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/Traverse/MatrixExample_forRows-2.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/Traverse/MatrixExample_forRows-2.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/Traverse/MatrixExample_forRows-2.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRows-1.cpp
 * \par Output
 * \include MatrixExample_forRows-1.out
 *
 * \include Matrices/Traverse/MatrixExample_forRows-2.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsWithIndexes.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsIf.cpp
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
 * \param matrix The matrix on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of row indexes
 *    whose corresponding rows will be processed using the lambda function.
 * \param rowCondition Lambda function to check row condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each row. See \ref TraversalRowFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Matrices/Traverse/MatrixExample_forRowsIf.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsIf.cpp
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
 * \include Matrices/Traverse/MatrixExample_forRowsIf.cpp
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

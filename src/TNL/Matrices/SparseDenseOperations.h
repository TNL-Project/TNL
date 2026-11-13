// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Matrices/DenseOperations.h>

namespace TNL::Matrices {

//! \brief The identity transformation of \ref DirectFetch, it returns the loaded value.
struct DirectFetchIdentity
{
   template< typename Value, typename Index >
   __cuda_callable__
   const Value&
   operator()( const Value& value, Index, Index ) const
   {
      return value;
   }
};

/**
 * \brief The access to a dense matrix stored in an array,
 * whose elements are transformed by a function before they are used.
 *
 * It can be passed as `fetchB` to \ref sparseDenseMatrixProductWithLambdas
 * instead of a lambda function of the form
 *
 * \code{.cpp}
 * fetchB = [=] __cuda_callable__ ( Index column, Index j ) { return transform( data[ column * stride + j ], column, j ); };
 * \endcode
 *
 * Since the product knows where the elements are read from,
 * it loads consecutive elements of the rows of row-major `B` by vectorized loads on GPUs
 * like for a dense matrix and applies `transform` to the loaded values.
 * A lambda function is called for each element separately,
 * which makes the product up to 2x slower for many columns of row-major `B`.
 * The values of a type narrower than the elements of `B` (e.g. half precision values transformed to `float`)
 * are loaded by vectorized loads of their bytes, so that less data is read from the memory.
 * Create it by \ref directFetch.
 *
 * \tparam Organization The organization of the array:
 *         \ref TNL::Algorithms::Segments::RowMajorOrder when the element in the row `column`
 *         and the column `j` is `data[ column * stride + j ]`,
 *         \ref TNL::Algorithms::Segments::ColumnMajorOrder when it is `data[ j * stride + column ]`.
 * \tparam Value The type of the elements of the array.
 * \tparam Index The type of the indexes.
 * \tparam Transform The type of the function `transform( const Value& value, Index column, Index j )`
 *         returning the element of the matrix.
 */
template<
   Algorithms::Segments::ElementsOrganization Organization,
   typename Value,
   typename Index,
   typename Transform = DirectFetchIdentity >
struct DirectFetch
{
   //! \brief The organization of the array.
   [[nodiscard]] static constexpr Algorithms::Segments::ElementsOrganization
   getOrganization()
   {
      return Organization;
   }

   //! \brief Returns the element in the row `column` and the column `j`.
   __cuda_callable__
   auto
   operator()( Index column, Index j ) const
   {
      return transform( data[ getIndex( column, j ) ], column, j );
   }

   //! \brief Returns the index of the element in the row `column` and the column `j` in the array.
   [[nodiscard]] __cuda_callable__
   Index
   getIndex( Index column, Index j ) const
   {
      if constexpr( Organization == Algorithms::Segments::RowMajorOrder )
         return column * stride + j;
      else
         return j * stride + column;
   }

   //! \brief Pointer to the array on the device of the sparse matrix.
   const Value* data;
   //! \brief The distance between the first elements of consecutive rows (row-major) or columns (column-major).
   Index stride;
   //! \brief The transformation of the loaded values.
   Transform transform;
};

/**
 * \brief Creates \ref DirectFetch reading the elements of a dense matrix from the array `data`.
 *
 * \tparam Organization The organization of the array, see \ref DirectFetch.
 * \param data Pointer to the array on the device of the sparse matrix.
 * \param stride The distance between the first elements of consecutive rows (row-major) or columns (column-major).
 * \param transform The function `transform( const Value& value, Index column, Index j )`
 *        applied to the loaded values.
 */
template<
   Algorithms::Segments::ElementsOrganization Organization,
   typename Value,
   typename Index,
   typename Transform = DirectFetchIdentity >
[[nodiscard]] DirectFetch< Organization, Value, Index, Transform >
directFetch( const Value* data, Index stride, Transform transform = {} )
{
   return { data, stride, transform };
}

//! \brief Tells whether `T` is \ref DirectFetch.
template< typename T >
struct IsDirectFetch : std::false_type
{};

template< Algorithms::Segments::ElementsOrganization Organization, typename Value, typename Index, typename Transform >
struct IsDirectFetch< DirectFetch< Organization, Value, Index, Transform > > : std::true_type
{};

//! \brief Tells whether `T` is \ref DirectFetch.
template< typename T >
constexpr bool is_direct_fetch_v = IsDirectFetch< std::decay_t< T > >::value;

/**
 * \brief Computes the product of a sparse and a dense matrix, `C = alpha * A * B + beta * C`.
 *
 * The columns of `B` and `C` are processed in tiles of at most `MaxTileSize` columns by
 * \ref TNL::Algorithms::Segments::multiReduceAllSegmentsInTiles over the rows of `A`, so every
 * element of `A` is read once per tile and used for all columns of the tile. It works with all
 * segment types and launch configurations of the sparse matrix and with both row-major and
 * column-major dense matrices. For one column it is equivalent to \ref SparseMatrixBase::vectorProduct.
 *
 * With the default launch configuration, a CSR matrix and a row-major `B` with more than 4 columns on GPU
 * (any number of columns on the host), the product is computed by \ref sparseDenseMatrixProductLanesOverColumns, which
 * is faster for many columns of `B`. The number of lanes and columns per lane are selected by
 * the number of columns of `B` and the precision and `MaxTileSize` is not used then. Otherwise, the default
 * configuration of CSR on GPU is selected by the average number of elements per row: the Tiered
 * mapping for less than 16 elements and the Uniform mapping with 16 threads per row for more.
 * On the host, the tile of column-major `B` is the smallest power of two not smaller than the number
 * of columns of `B` (at most `MaxTileSize`), and it is reduced down to 4 columns when the columns
 * of `B` of one tile do not fit in the L3 cache.
 *
 * Symmetric sparse matrices are not supported yet.
 *
 * \tparam MaxTileSize The maximal number of columns of `B` and `C` processed at once.
 * \tparam SparseMatrix The type of the sparse matrix `A` (\ref SparseMatrix or \ref SparseMatrixView).
 * \tparam DenseMatrixB The type of the dense matrix `B` (\ref DenseMatrix or \ref DenseMatrixView).
 * \tparam DenseMatrixC The type of the dense matrix `C` (\ref DenseMatrix or \ref DenseMatrixView).
 * \tparam Real The type of the multiplicators.
 *
 * \param A The sparse matrix with `m` rows and `n` columns.
 * \param B The dense matrix with `n` rows and `k` columns.
 * \param C The dense matrix with `m` rows and `k` columns, the result is stored here.
 * \param alpha The multiplicator of the product `A * B`.
 * \param beta The multiplicator of the original values of `C`. If it is zero, the original
 *             values of `C` are not read.
 * \param launchConfig The launch configuration of the reduction in the segments of `A`.
 */
template<
   int MaxTileSize = 16,
   typename SparseMatrix,
   typename DenseMatrixB,
   typename DenseMatrixC,
   typename Real = typename SparseMatrix::ComputeRealType >
void
sparseDenseMatrixProduct(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha = 1,
   Real beta = 0,
   const Algorithms::Segments::LaunchConfiguration& launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes the product of a sparse matrix and a dense matrix or its transposition,
 * `C = alpha * A * op( B ) + beta * C`.
 *
 * `op( B )` is `B` or its transposition `B^T` given by `transposeB`. The transposition is not
 * computed, the elements of `B` are read in the transposed order on the fly, so the product with
 * `B^T` is computed like the product with a matrix of the opposite organization: column-major `B`
 * like row-major `B^T` (its consecutive columns are consecutive in memory and they are read by
 * vectorized loads) and row-major `B` like column-major `B^T`. See \ref sparseDenseMatrixProduct
 * for the details and the other parameters.
 *
 * \param B The dense matrix with `n` rows and `k` columns (`k` rows and `n` columns for
 *        \ref TransposeState::Transpose).
 * \param transposeB Whether the product is computed with `B` or with its transposition.
 */
template<
   int MaxTileSize = 16,
   typename SparseMatrix,
   typename DenseMatrixB,
   typename DenseMatrixC,
   typename Real = typename SparseMatrix::ComputeRealType >
void
sparseDenseMatrixProduct(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha,
   Real beta,
   TransposeState transposeB,
   const Algorithms::Segments::LaunchConfiguration& launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes the product of a sparse and a dense matrix, `C = alpha * A * B + beta * C`,
 * with the threads of each row of `A` distributed over the columns of `B` and `C`.
 *
 * The product is computed by one \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
 * On GPUs, the columns of `B` and `C` are split into tiles of `Lanes * ColumnsPerLane` columns,
 * a group of `Lanes` threads processes every row of `A` and tile and each thread computes
 * `ColumnsPerLane` consecutive columns of the tile. Hence for row-major `B` the threads of the group
 * read consecutive elements of its rows and no reduction between the threads is needed. All tiles
 * are processed at once, so the product is efficient also for a few rows of `A` and many columns
 * of `B`.
 *
 * The rows of `A` with more than `maxRowLength` elements are split into pieces processed by
 * whole thread blocks, see \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
 *
 * The threads are distributed over the columns only for the CSR format of `A` on GPUs, otherwise
 * the product is computed by the general multireduction, see
 * \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
 * See \ref sparseDenseMatrixProduct for the other parameters.
 *
 * \tparam Lanes The number of threads per row of `A`, a power of two up to the warp size.
 * \tparam ColumnsPerLane The number of consecutive columns computed by each thread.
 * \param maxRowLength The maximal number of elements of the rows processed by one group of
 *        threads, zero means the default.
 */
template<
   int Lanes,
   int ColumnsPerLane,
   typename SparseMatrix,
   typename DenseMatrixB,
   typename DenseMatrixC,
   typename Real = typename SparseMatrix::ComputeRealType >
void
sparseDenseMatrixProductLanesOverColumns(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha = 1,
   Real beta = 0,
   typename SparseMatrix::IndexType maxRowLength = 0 );

/**
 * \brief Computes the product of a sparse matrix and a dense matrix given by lambda functions.
 *
 * It computes `C = A * B`, where the elements of the dense matrix `B` with `n` rows and `k`
 * columns are returned by `fetchB( column, j )` and every element of the result is passed to
 * `storeC( row, j, value )`. Hence `B` can be computed on the fly or read from any data
 * structure and `C` can be post-processed before it is stored, without auxiliary dense matrices.
 * The product is computed by the same kernels as \ref sparseDenseMatrixProduct, the kernel and
 * the tiles of columns are selected in the same way by `Organization`, which describes how
 * `fetchB` reads the data: \ref TNL::Algorithms::Segments::RowMajorOrder when the elements
 * for consecutive `j` are consecutive in memory, \ref TNL::Algorithms::Segments::ColumnMajorOrder
 * when the elements for consecutive `column` are. The only difference is that the elements of `B`
 * are not read by vectorized loads, which makes the product up to 2x slower on GPUs when the loads
 * of `B` from the cache limit it (many columns of row-major `B` and long rows of `A`). Therefore
 * `fetchB` can also be a dense matrix or a dense matrix view, which is then read like in
 * \ref sparseDenseMatrixProduct and `Organization` is not used, and only `C` is given by a lambda
 * function. When the elements of `B` are read from an array and only transformed,
 * `fetchB` should be \ref DirectFetch created by \ref directFetch,
 * which keeps the vectorized loads; `Organization` is then given by it.
 *
 * Both functions must be callable on the device of `A` and `storeC` is called exactly once for
 * every `row` and `j`.
 *
 * \tparam Organization How `fetchB` accesses the memory, it selects the kernel.
 * \tparam MaxTileSize The maximal number of columns of `B` processed at once.
 * \tparam SparseMatrix The type of the sparse matrix `A` (\ref SparseMatrix or \ref SparseMatrixView).
 * \tparam FetchB The type of the function `fetchB( IndexType column, IndexType j ) -> Real`.
 * \tparam StoreC The type of the function `storeC( IndexType row, int j, const ComputeRealType& value )`.
 *
 * \param A The sparse matrix with `m` rows and `n` columns.
 * \param k The number of columns of `B` and `C`.
 * \param fetchB The function returning the element of `B` in the row `column` and the column `j`,
 *        \ref DirectFetch or the dense matrix `B`.
 * \param storeC The function storing the element of `C` in the row `row` and the column `j`.
 * \param launchConfig The launch configuration of the reduction in the segments of `A`.
 */
template<
   Algorithms::Segments::ElementsOrganization Organization = Algorithms::Segments::RowMajorOrder,
   int MaxTileSize = 16,
   typename SparseMatrix,
   typename FetchB,
   typename StoreC >
void
sparseDenseMatrixProductWithLambdas(
   const SparseMatrix& A,
   typename SparseMatrix::IndexType k,
   FetchB&& fetchB,
   StoreC&& storeC,
   const Algorithms::Segments::LaunchConfiguration& launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Computes the product of a sparse matrix and a dense matrix given by lambda functions
 * with the threads of each row of `A` distributed over the columns of `B` and `C`.
 *
 * It is \ref sparseDenseMatrixProductWithLambdas computed by the kernel of
 * \ref sparseDenseMatrixProductLanesOverColumns with the given number of lanes and columns per lane,
 * e.g. when the best configuration for the given data is selected by measurement.
 *
 * \tparam Organization How `fetchB` accesses the memory when it is a lambda function,
 *         see \ref sparseDenseMatrixProductWithLambdas.
 * \tparam Lanes The number of threads per row of `A`, a power of two up to the warp size.
 * \tparam ColumnsPerLane The number of consecutive columns computed by each thread.
 * \param A The sparse matrix with `m` rows and `n` columns.
 * \param k The number of columns of `B` and `C`.
 * \param fetchB The function returning the element of `B` in the row `column` and the column `j`,
 *        \ref DirectFetch or the dense matrix `B`.
 * \param storeC The function storing the element of `C` in the row `row` and the column `j`.
 * \param maxRowLength The maximal number of elements of the rows processed by one group of
 *        threads, zero means the default.
 */
template<
   int Lanes,
   int ColumnsPerLane,
   Algorithms::Segments::ElementsOrganization Organization = Algorithms::Segments::RowMajorOrder,
   typename SparseMatrix,
   typename FetchB,
   typename StoreC >
void
sparseDenseMatrixProductWithLambdasLanesOverColumns(
   const SparseMatrix& A,
   typename SparseMatrix::IndexType k,
   FetchB&& fetchB,
   StoreC&& storeC,
   typename SparseMatrix::IndexType maxRowLength = 0 );

/**
 * \brief Computes the product of a dense and a sparse matrix, `C = alpha * B * op( A ) + beta * C`.
 *
 * `op( A )` is `A` or its transposition `A^T` given by `transposeA`. The rows of the sparse matrix are its segments,
 * so the two cases are computed differently:
 *
 * - \ref TransposeState::Transpose: the element `C( i, j )` is the dot product of the row `i` of `B` and the row `j`
 *   of `A`, i.e. `C^T = A * B^T` is computed by the reduction in the rows of `A` like \ref sparseDenseMatrixProduct,
 *   without the transpositions of `B` and `C`, which are only read and written in the transposed order. The rows of
 *   `B` (`k` columns of `B^T`) are processed in tiles of at most `MaxTileSize` rows. Column-major `B` and `C` are read
 *   and written by consecutive elements (and `B` by vectorized loads), so they are more efficient than row-major ones.
 *   Hence, when the product with a sparse matrix `A` stored by rows is computed repeatedly, it is efficient to store
 *   `A^T` and compute the product with its transposition.
 * - \ref TransposeState::None: the row `j` of `A` contributes `B( :, j ) * A( j, l )` to the columns `l` of `C` given
 *   by its elements. Every row of `A` and row of `B` is processed by one thread, which adds the contributions by atomic
 *   additions. For the types without atomic additions (e.g. complex numbers), `A` (which must be \ref SparseMatrix then)
 *   is transposed and the product is computed as for \ref TransposeState::Transpose.
 *
 * Symmetric sparse matrices are not supported yet.
 *
 * \tparam MaxTileSize The maximal number of rows of `B` and `C` processed at once for \ref TransposeState::Transpose.
 * \tparam DenseMatrixB The type of the dense matrix `B` (\ref DenseMatrix or \ref DenseMatrixView).
 * \tparam SparseMatrix The type of the sparse matrix `A` (\ref SparseMatrix or \ref SparseMatrixView).
 * \tparam DenseMatrixC The type of the dense matrix `C` (\ref DenseMatrix or \ref DenseMatrixView).
 * \tparam Real The type of the multiplicators.
 *
 * \param B The dense matrix with `k` rows and `n` columns.
 * \param A The sparse matrix with `n` rows and `m` columns (`m` rows and `n` columns for
 *        \ref TransposeState::Transpose).
 * \param C The dense matrix with `k` rows and `m` columns, the result is stored here.
 * \param alpha The multiplicator of the product `B * op( A )`.
 * \param beta The multiplicator of the original values of `C`. If it is zero, the original values of `C` are not read.
 * \param transposeA Whether the product is computed with `A` or with its transposition.
 * \param launchConfig The launch configuration of the reduction in the segments of `A` for
 *        \ref TransposeState::Transpose.
 */
template<
   int MaxTileSize = 16,
   typename DenseMatrixB,
   typename SparseMatrix,
   typename DenseMatrixC,
   typename Real = typename SparseMatrix::ComputeRealType >
void
denseSparseMatrixProduct(
   const DenseMatrixB& B,
   const SparseMatrix& A,
   DenseMatrixC& C,
   Real alpha = 1,
   Real beta = 0,
   TransposeState transposeA = TransposeState::None,
   const Algorithms::Segments::LaunchConfiguration& launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Matrices

#include "SparseDenseOperations.hpp"

// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Algorithms/Segments/CSR.h>

#include "SparseMatrixBase.h"
#include "MatrixType.h"

namespace TNL::Matrices {

// TODO: move this into the detail namespace (which is implicitly hidden in the doc)
/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Real, typename Index = int >
struct ChooseSparseMatrixComputeReal
{
   using type = Real;
};

template< typename Index >
struct ChooseSparseMatrixComputeReal< bool, Index >
{
   using type = Index;
};
/// \endcond

/**
 * \brief Implementation of sparse matrix view.
 *
 * It serves as an accessor to \ref SparseMatrix for example when passing the
 * matrix to lambda functions. SparseMatrix view can be also created in CUDA
 * kernels.
 *
 * \tparam Real is a type of matrix elements. If \e Real equals \e bool the
 *         matrix is treated as binary and so the matrix elements values are
 *         not stored in the memory since we need to remember only coordinates
 *         of non-zero elements (which equal one).
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam MatrixType specifies a symmetry of matrix. See \ref MatrixType.
 *         Symmetric matrices store only lower part of the matrix and its
 *         diagonal. The upper part is reconstructed on the fly.  GeneralMatrix
 *         with no symmetry is used by default.
 * \tparam Segments is a structure representing the sparse matrix format.
 *         Depending on the pattern of the non-zero elements different matrix
 *         formats can perform differently especially on GPUs. By default
 *         \ref Algorithms::Segments::CSR format is used. See also
 *         \ref Algorithms::Segments::Ellpack,
 *         \ref Algorithms::Segments::SlicedEllpack,
 *         \ref Algorithms::Segments::ChunkedEllpack, and
 *         \ref Algorithms::Segments::BiEllpack.
 * \tparam ComputeReal is the same as \e Real mostly but for binary matrices it
 *         is set to \e Index type. This can be changed by the user, of course.
 */
template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename MatrixType_ = GeneralMatrix,
          template< typename Device_, typename Index_ > class SegmentsView = Algorithms::Segments::CSRView,
          typename ComputeReal = typename ChooseSparseMatrixComputeReal< Real, Index >::type >
class SparseMatrixView : public SparseMatrixBase< Real,
                                                  Device,
                                                  Index,
                                                  MatrixType_,
                                                  std::conditional_t< std::is_const_v< Real >,
                                                                      typename SegmentsView< Device, Index >::ConstViewType,
                                                                      SegmentsView< Device, Index > >,
                                                  ComputeReal >
{
   using Base = SparseMatrixBase< Real,
                                  Device,
                                  Index,
                                  MatrixType_,
                                  std::conditional_t< std::is_const_v< Real >,
                                                      typename SegmentsView< Device, Index >::ConstViewType,
                                                      SegmentsView< Device, Index > >,
                                  ComputeReal >;

public:
   using Base::MatrixType;
   /**
    * \brief Templated type of segments view, i.e. sparse matrix format.
    */
   template< typename Device_, typename Index_ >
   using SegmentsViewTemplate = SegmentsView< Device_, Index_ >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             typename _MatrixType = MatrixType_,
             template< typename, typename > class _SegmentsView = SegmentsViewTemplate,
             typename _ComputeReal = ComputeReal >
   using Self = SparseMatrixView< _Real, _Device, _Index, _MatrixType, _SegmentsView, _ComputeReal >;

   /**
    * \brief Type of related matrix view.
    */
   using ViewType = SparseMatrixView< Real, Device, Index, MatrixType_, SegmentsViewTemplate, ComputeReal >;

   /**
    * \brief Matrix view type for constant instances.
    */
   using ConstViewType =
      SparseMatrixView< std::add_const_t< Real >, Device, Index, MatrixType_, SegmentsViewTemplate, ComputeReal >;

   /**
    * \brief Constructor with no parameters.
    */
   __cuda_callable__
   SparseMatrixView() = default;

   /**
    * \brief Constructor with all necessary data and views.
    *
    * \param rows is a number of matrix rows.
    * \param columns is a number of matrix columns.
    * \param values is a vector view with matrix elements values.
    * \param columnIndexes is a vector view with matrix elements column indexes.
    * \param segments is a segments view representing the sparse matrix format.
    */
   __cuda_callable__
   SparseMatrixView( Index rows,
                     Index columns,
                     typename Base::ValuesViewType values,
                     typename Base::ColumnIndexesViewType columnIndexes,
                     typename Base::SegmentsViewType segments );

   /**
    * \brief Copy constructor.
    *
    * \param matrix is an input sparse matrix view.
    */
   __cuda_callable__
   SparseMatrixView( const SparseMatrixView& matrix ) = default;

   /**
    * \brief Move constructor.
    *
    * \param matrix is an input sparse matrix view.
    */
   __cuda_callable__
   SparseMatrixView( SparseMatrixView&& matrix ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because sparse matrix assignment in general
    * requires reallocation.
    */
   SparseMatrixView&
   operator=( const SparseMatrixView& ) = delete;

   /**
    * \brief Method for rebinding (reinitialization) using another sparse matrix view.
    *
    * \param view The sparse matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( SparseMatrixView& view );

   /**
    * \brief Method for rebinding (reinitialization) using another sparse matrix view.
    *
    * \param view The sparse matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( SparseMatrixView&& view );

   /**
    * \brief Returns a modifiable view of the sparse matrix.
    *
    * \return sparse matrix view.
    */
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the sparse matrix.
    *
    * \return sparse matrix view.
    */
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;
};

/**
 * \brief Deserialization of sparse matrix views from binary files.
 */
template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
File&
operator>>( File& file, SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >& matrix );

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename > class SegmentsView,
          typename ComputeReal >
File&
operator>>( File&& file, SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >& matrix );

}  // namespace TNL::Matrices

#include "SparseMatrixView.hpp"

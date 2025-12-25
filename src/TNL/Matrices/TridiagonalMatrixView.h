// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "TridiagonalMatrixBase.h"

namespace TNL::Matrices {

/**
 * \brief Implementation of sparse tridiagonal matrix.
 *
 * It serves as an accessor to \ref SparseMatrix for example when passing the
 * matrix to lambda functions. SparseMatrix view can be also created in CUDA kernels.
 *
 * See \ref TridiagonalMatrix for more details.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class TridiagonalMatrixView : public TridiagonalMatrixBase< Real, Device, Index, Organization >
{
   using Base = TridiagonalMatrixBase< Real, Device, Index, Organization >;

public:
   /**
    */
   using ViewType = TridiagonalMatrixView< Real, Device, Index, Organization >;

   /**
    * \brief Matrix view type for constant instances.
    */
   using ConstViewType = TridiagonalMatrixView< std::add_const_t< Real >, Device, Index, Organization >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             ElementsOrganization Organization_ =
                Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
   using Self = TridiagonalMatrixView< _Real, _Device, _Index, Organization_ >;

   /**
    * \brief Constructor with no parameters.
    */
   __cuda_callable__
   TridiagonalMatrixView() = default;

   /**
    * \brief Constructor with all necessary data and views.
    *
    * \param values is a vector view with matrix elements values
    * \param indexer is an indexer of matrix elements
    */
   __cuda_callable__
   TridiagonalMatrixView( typename Base::ValuesViewType values, typename Base::IndexerType indexer );

   /**
    * \brief Copy constructor.
    */
   __cuda_callable__
   TridiagonalMatrixView( const TridiagonalMatrixView& ) = default;

   /**
    * \brief Move constructor.
    */
   __cuda_callable__
   TridiagonalMatrixView( TridiagonalMatrixView&& ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because matrix assignment in general requires
    * reallocation.
    */
   TridiagonalMatrixView&
   operator=( const TridiagonalMatrixView& ) = delete;

   /**
    * \brief Move-assignment operator.
    */
   TridiagonalMatrixView&
   operator=( TridiagonalMatrixView&& ) = delete;

   /**
    * \brief Method for rebinding (reinitialization) using another tridiagonal matrix view.
    *
    * \param view The tridiagonal matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( TridiagonalMatrixView view );

   /**
    * \brief Returns a modifiable view of the tridiagonal matrix.
    *
    * \return tridiagonal matrix view.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the tridiagonal matrix.
    *
    * \return tridiagonal matrix view.
    */
   [[nodiscard]] ConstViewType
   getConstView() const;
};

/**
 * \brief Deserialization of tridiagonal matrix views from binary files.
 */
template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File& file, TridiagonalMatrixView< Real, Device, Index, Organization >& matrix );

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator>>( File&& file, TridiagonalMatrixView< Real, Device, Index, Organization >& matrix );

}  // namespace TNL::Matrices

#include "TridiagonalMatrixView.hpp"

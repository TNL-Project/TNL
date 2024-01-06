// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "MultidiagonalMatrixBase.h"

namespace TNL::Matrices {

/**
 * \brief Implementation of sparse multidiagonal matrix.
 *
 * It serves as an accessor to \ref SparseMatrix for example when passing the
 * matrix to lambda functions. SparseMatrix view can be also created in CUDA kernels.
 *
 * See \ref MultidiagonalMatrix for more details.
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
class MultidiagonalMatrixView : public MultidiagonalMatrixBase< Real, Device, Index, Organization >
{
   using Base = MultidiagonalMatrixBase< Real, Device, Index, Organization >;

public:
   /**
    * \brief The type of matrix elements.
    */
   using RealType = typename Base::RealType;

   /**
    * \brief The device where the matrix is allocated.
    */
   using DeviceType = Device;

   /**
    * \brief The type used for matrix elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Type of related matrix view.
    */
   using ViewType = MultidiagonalMatrixView< Real, Device, Index, Organization >;

   /**
    * \brief Matrix view type for constant instances.
    */
   using ConstViewType = MultidiagonalMatrixView< std::add_const_t< Real >, Device, Index, Organization >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             ElementsOrganization Organization_ =
                Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
   using Self = MultidiagonalMatrixView< _Real, _Device, _Index, Organization_ >;

   /**
    * \brief Constructor with no parameters.
    */
   __cuda_callable__
   MultidiagonalMatrixView() = default;

   /**
    * \brief Constructor with all necessary data and views.
    *
    * \param values is a vector view with matrix elements values
    * \param diagonalOffsets is a vector view with diagonals offsets
    * \param hostDiagonalOffsets is a vector view with a copy of diagonals offsets on the host
    * \param indexer is an indexer of matrix elements
    */
   __cuda_callable__
   MultidiagonalMatrixView( typename Base::ValuesViewType values,
                            typename Base::DiagonalOffsetsView diagonalOffsets,
                            typename Base::HostDiagonalOffsetsView hostDiagonalOffsets,
                            typename Base::IndexerType indexer );

   /**
    * \brief Copy constructor.
    */
   __cuda_callable__
   MultidiagonalMatrixView( const MultidiagonalMatrixView& ) = default;

   /**
    * \brief Move constructor.
    */
   __cuda_callable__
   MultidiagonalMatrixView( MultidiagonalMatrixView&& ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because matrix assignment in general requires
    * reallocation.
    */
   MultidiagonalMatrixView&
   operator=( const MultidiagonalMatrixView& ) = delete;

   /**
    * \brief Move-assignment operator.
    */
   MultidiagonalMatrixView&
   operator=( MultidiagonalMatrixView&& ) = delete;

   /**
    * \brief Method for rebinding (reinitialization) using another multidiagonal matrix view.
    *
    * \param view The multidiagonal matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( MultidiagonalMatrixView view );

   /**
    * \brief Returns a modifiable view of the multidiagonal matrix.
    *
    * \return multidiagonal matrix view.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the multidiagonal matrix.
    *
    * \return multidiagonal matrix view.
    */
   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Method for saving the matrix to a file.
    *
    * \param file is the output file.
    */
   void
   save( File& file ) const;
};

}  // namespace TNL::Matrices

#include "MultidiagonalMatrixView.hpp"

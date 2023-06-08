// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>
#include "DenseMatrixBase.h"

namespace TNL::Matrices {

/**
 * \brief Implementation of dense matrix view.
 *
 * It serves as an accessor to \ref DenseMatrix for example when passing the
 * matrix to lambda functions. DenseMatrix view can be also created in CUDA kernels.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam MatrixElementsOrganization tells the ordering of matrix elements in memory. It is either
 *         \ref TNL::Algorithms::Segments::RowMajorOrder
 *         or \ref TNL::Algorithms::Segments::ColumnMajorOrder.
 *
 * See \ref DenseMatrix.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class DenseMatrixView : public DenseMatrixBase< Real, Device, Index, Organization >
{
   using Base = DenseMatrixBase< Real, Device, Index, Organization >;

public:
   /**
    * \brief Matrix view type.
    *
    * See \ref DenseMatrixView.
    */
   using ViewType = DenseMatrixView< Real, Device, Index, Organization >;

   /**
    * \brief Matrix view type for constant instances.
    *
    * See \ref DenseMatrixView.
    */
   using ConstViewType = DenseMatrixView< std::add_const_t< Real >, Device, Index, Organization >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real, typename _Device = Device, typename _Index = Index >
   using Self = DenseMatrixView< _Real, _Device, _Index >;

   /**
    * \brief Constructor without parameters.
    */
   __cuda_callable__
   DenseMatrixView() = default;

   /**
    * \brief Constructor with matrix dimensions and values.
    *
    * Organization of matrix elements values in
    *
    * \param rows number of matrix rows.
    * \param columns number of matrix columns.
    * \param values is vector view with matrix elements values.
    *
    * \par Example
    * \include Matrices/DenseMatrix/DenseMatrixViewExample_constructor.cpp
    * \par Output
    * \include DenseMatrixViewExample_constructor.out
    */
   __cuda_callable__
   DenseMatrixView( Index rows, Index columns, typename Base::ValuesViewType values );

   /**
    * \brief Copy constructor.
    *
    * \param matrix is the source matrix view.
    */
   __cuda_callable__
   DenseMatrixView( const DenseMatrixView& matrix ) = default;

   /**
    * \brief Move constructor.
    *
    * \param matrix is the source matrix view.
    */
   __cuda_callable__
   DenseMatrixView( DenseMatrixView&& matrix ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because matrix assignment in general requires
    * reallocation.
    */
   DenseMatrixView&
   operator=( const DenseMatrixView& ) = delete;

   /**
    * \brief Move-assignment operator.
    */
   DenseMatrixView&
   operator=( DenseMatrixView&& ) = delete;

   /**
    * \brief Method for rebinding (reinitialization) using another dense matrix view.
    *
    * \param view The dense matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( DenseMatrixView& view );

   /**
    * \brief Method for rebinding (reinitialization) using another dense matrix view.
    *
    * \param view The dense matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( DenseMatrixView&& view );

   /**
    * \brief Returns a modifiable dense matrix view.
    *
    * \return dense matrix view.
    */
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   /**
    * \brief Returns a non-modifiable dense matrix view.
    *
    * \return dense matrix view.
    */
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;

   /**
    * \brief Method for saving the matrix view to a file.
    *
    * The ouput file can be loaded by \ref DenseMatrix.
    *
    * \param file is the file where the matrix will be saved.
    */
   void
   save( File& file ) const;
};

}  // namespace TNL::Matrices

#include "DenseMatrixView.hpp"

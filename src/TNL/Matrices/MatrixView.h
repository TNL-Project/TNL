// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "MatrixBase.h"

namespace TNL::Matrices {

/**
 * \brief Base class for other matrix types views.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 */
template< typename Real, typename Device, typename Index >
class [[deprecated]] MatrixView : public MatrixBase< Real, Device, Index, GeneralMatrix, Algorithms::Segments::RowMajorOrder >
{
   using Base = MatrixBase< Real, Device, Index, GeneralMatrix, Algorithms::Segments::RowMajorOrder >;

public:
   /**
    * \brief Basic constructor with no parameters.
    */
   __cuda_callable__
   MatrixView() = default;

   /**
    * \brief Constructor with matrix dimensions and matrix elements values.
    *
    * The matrix elements values are passed in a form vector view.
    *
    * @param rows is a number of matrix rows.
    * @param columns is a number of matrix columns.
    * @param values is a vector view with matrix elements values.
    */
   __cuda_callable__
   MatrixView( Index rows, Index columns, typename Base::ValuesViewType values );

   /**
    * @brief Copy constructor.
    *
    * @param view is an input matrix view.
    */
   __cuda_callable__
   MatrixView( const MatrixView& view ) = default;

   /**
    * \brief Move constructor.
    *
    * @param view is an input matrix view.
    */
   __cuda_callable__
   MatrixView( MatrixView&& view ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because matrix assignment in general requires
    * reallocation.
    */
   __cuda_callable__
   MatrixView&
   operator=( const MatrixView& ) = delete;

   /**
    * \brief Move-assignment operator.
    */
   __cuda_callable__
   MatrixView&
   operator=( MatrixView&& ) = delete;

   /**
    * \brief Method for rebinding (reinitialization) using another matrix view.
    *
    * \param view The matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( MatrixView& view );

   /**
    * \brief Method for rebinding (reinitialization) using another matrix view.
    *
    * \param view The matrix view to be bound.
    */
   __cuda_callable__
   void
   bind( MatrixView&& view );

   /***
    * \brief Virtual serialization type getter.
    *
    * Objects in TNL are saved as in a device independent manner. This method
    * is supposed to return the object type but with the device type replaced
    * by Devices::Host. For example \c Array< double, Devices::Cuda > is
    * saved as \c Array< double, Devices::Host >.
    */
   [[nodiscard]] virtual std::string
   getSerializationTypeVirtual() const = 0;

   /**
    * \brief Method for saving the matrix view to a file.
    *
    * \param file is the output file.
    */
   virtual void
   save( File& file ) const;

   /**
    * \brief Method for saving the matrix view to a file.
    *
    * \param fileName String defining the name of a file.
    */
   void
   save( const String& fileName ) const;

   /**
    * \brief Method for printing the matrix view to output stream.
    *
    * \param str is the output stream.
    */
   virtual void
   print( std::ostream& str ) const;
};

/**
 * \brief Overloaded insertion operator for printing a matrix to output stream.
 *
 * \tparam Real is a type of the matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type used for the indexing of the matrix elements.
 *
 * \param str is a output stream.
 * \param matrix is the matrix to be printed.
 *
 * \return a reference to the output stream \ref std::ostream.
 */
template< typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const MatrixView< Real, Device, Index >& matrix )
{
   matrix.print( str );
   return str;
}

}  // namespace TNL::Matrices

#include <TNL/Matrices/MatrixView.hpp>

// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Matrices {

/**
 * \brief Accessor for dense matrix elements.
 *
 * \tparam Real is a type of matrix elements values.
 * \tparam Index is a type of matrix elements column indexes.
 */
template< typename Real, typename Index >
class DenseMatrixElement
{
public:
   /**
    * \brief Type of matrix elements values.
    */
   using RealType = Real;

   /**
    * \brief Type of matrix elements column indexes.
    */
   using IndexType = Index;

   /**
    * \brief Constructor.
    *
    * \param value is matrix element value.
    * \param rowIdx is row index of the matrix element.
    * \param columnIdx is a column index of the matrix element.
    * \param localIdx is the column index of the matrix element as well.
    */
   __cuda_callable__
   DenseMatrixElement( RealType& value,
                       IndexType rowIdx,
                       IndexType columnIdx,
                       IndexType localIdx )  // localIdx is here only for compatibility with SparseMatrixElement
   : value_( value ),
     rowIdx( rowIdx ),
     columnIdx( columnIdx )
   {}

   /**
    * \brief Returns reference on matrix element value.
    *
    * \return reference on matrix element value.
    */
   [[nodiscard]] __cuda_callable__
   RealType&
   value()
   {
      return value_;
   }

   /**
    * \brief Returns constant reference on matrix element value.
    *
    * \return constant reference on matrix element value.
    */
   [[nodiscard]] __cuda_callable__
   const RealType&
   value() const
   {
      return value_;
   }

   /**
    * \brief Returns the row index of the matrix element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   rowIndex() const
   {
      return rowIdx;
   }

   /**
    * \brief Returns the column index of the matrix element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   columnIndex() const
   {
      return columnIdx;
   }

   /**
    * \brief Returns the column index of the matrix element.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   localIndex() const
   {
      return columnIdx;
   }

protected:
   RealType& value_;

   const IndexType rowIdx;

   const IndexType columnIdx;
};

}  // namespace TNL::Matrices

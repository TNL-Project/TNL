// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixRowViewIterator.h>
#include <TNL/Matrices/MultidiagonalMatrixElement.h>

namespace TNL::Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of tridiagonal matrix.
 *
 * \tparam ValuesView is a vector view storing the matrix elements values.
 * \tparam Indexer is type of object responsible for indexing and organization of
 *    matrix elements.
 *
 * See \ref TridiagonalMatrix and \ref TridiagonalMatrixView.
 *
 * \par Example
 * \include Matrices/TridiagonalMatrix/TridiagonalMatrixViewExample_getRow.cpp
 * \par Output
 * \include TridiagonalMatrixViewExample_getRow.out
 */
template< typename ValuesView, typename Indexer >
class TridiagonalMatrixRowView
{
public:
   /**
    * \brief The type of matrix elements.
    */
   using RealType = typename ValuesView::RealType;

   /**
    * \brief The type used for matrix elements indexing.
    */
   using IndexType = typename ValuesView::IndexType;

   /**
    * \brief Type of container view used for storing the matrix elements values.
    */
   using ValuesViewType = ValuesView;

   /**
    * \brief Type of object responsible for indexing and organization of
    * matrix elements.
    */
   using IndexerType = Indexer;

   /**
    * \brief Type of constant container view used for storing the matrix elements values.
    */
   using ConstValuesViewType = typename ValuesViewType::ConstViewType;

   /**
    * \brief Type of constant sparse matrix row view.
    */
   using RowView = TridiagonalMatrixRowView< ValuesViewType, IndexerType >;

   /**
    * \brief Type of constant sparse matrix row view.
    */
   using ConstRowView = TridiagonalMatrixRowView< ConstValuesViewType, IndexerType >;

   /**
    * \brief The type of related matrix element.
    */
   using MatrixElementType = MultidiagonalMatrixElement< RealType, IndexType >;

   /**
    * \brief Type of iterator for the matrix row.
    */
   using IteratorType = MatrixRowViewIterator< RowView >;

   /**
    * \brief Type of constant iterator for the matrix row.
    */
   using ConstIteratorType = MatrixRowViewIterator< ConstRowView >;

   /**
    * \brief Constructor with all necessary data.
    *
    * \param rowIdx is index of the matrix row this RowView refer to.
    * \param values is a vector view holding values of matrix elements.
    * \param indexer is object responsible for indexing and organization of matrix elements
    */
   __cuda_callable__
   TridiagonalMatrixRowView( IndexType rowIdx, const ValuesViewType& values, const IndexerType& indexer );

   /**
    * \brief Returns number of diagonals of the tridiagonal matrix which is three.
    *
    * \return number three.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getSize() const;

   /**
    * \brief Returns the matrix row index.
    *
    * \return matrix row index.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getRowIndex() const;

   /**
    * \brief Computes column index of matrix element on given subdiagonal.
    *
    * \param localIdx is an index of the subdiagonal.
    *
    * \return column index of matrix element on given subdiagonal.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getColumnIndex( IndexType localIdx ) const;

   /**
    * \brief Returns value of matrix element on given subdiagonal.
    *
    * \param localIdx is an index of the subdiagonal.
    *
    * \return constant reference to matrix element value.
    */
   [[nodiscard]] __cuda_callable__
   const RealType&
   getValue( IndexType localIdx ) const;

   /**
    * \brief Returns value of matrix element on given subdiagonal.
    *
    * \param localIdx is an index of the subdiagonal.
    *
    * \return non-constant reference to matrix element value.
    */
   [[nodiscard]] __cuda_callable__
   RealType&
   getValue( IndexType localIdx );

   /**
    * \brief Changes value of matrix element on given subdiagonal.
    *
    * \param localIdx is an index of the matrix subdiagonal.
    * \param value is the new value of the matrix element.
    */
   __cuda_callable__
   void
   setElement( IndexType localIdx, const RealType& value );

   /**
    * \brief Returns iterator pointing at the beginning of the matrix row.
    *
    * \return iterator pointing at the beginning.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   begin();

   /**
    * \brief Returns iterator pointing at the end of the matrix row.
    *
    * \return iterator pointing at the end.
    */
   [[nodiscard]] __cuda_callable__
   IteratorType
   end();

   /**
    * \brief Returns constant iterator pointing at the beginning of the matrix row.
    *
    * \return iterator pointing at the beginning.
    */
   [[nodiscard]] __cuda_callable__
   ConstIteratorType
   cbegin() const;

   /**
    * \brief Returns constant iterator pointing at the end of the matrix row.
    *
    * \return iterator pointing at the end.
    */
   [[nodiscard]] __cuda_callable__
   ConstIteratorType
   cend() const;

protected:
   IndexType rowIdx;

   ValuesViewType values;

   Indexer indexer;
};

}  // namespace TNL::Matrices

#include <TNL/Matrices/TridiagonalMatrixRowView.hpp>

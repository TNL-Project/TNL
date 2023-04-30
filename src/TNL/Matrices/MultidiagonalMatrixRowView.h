// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MultidiagonalMatrixElement.h>
#include <TNL/Matrices/MatrixRowViewIterator.h>

namespace TNL::Matrices {

/**
 * \brief RowView is a simple structure for accessing rows of multidiagonal matrix.
 *
 * \tparam ValuesView is a vector view storing the matrix elements values.
 * \tparam Indexer is type of object responsible for indexing and organization of
 *    matrix elements.
 * \tparam DiagonalsOffsetsView_ is a container view holding offsets of
 *    diagonals of multidiagonal matrix.
 *
 * See \ref MultidiagonalMatrix and \ref MultidiagonalMatrixView.
 *
 * \par Example
 * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_getRow.cpp
 * \par Output
 * \include MultidiagonalMatrixExample_getRow.out
 *
 * \par Example
 * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getRow.cpp
 * \par Output
 * \include MultidiagonalMatrixViewExample_getRow.out
 */
template< typename ValuesView, typename Indexer, typename DiagonalsOffsetsView_ >
class MultidiagonalMatrixRowView
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
    * \brief Type of a container view holding offsets of
    * diagonals of multidiagonal matrix.
    */
   using DiagonalsOffsetsView = DiagonalsOffsetsView_;

   /**
    * \brief Type of constant container view used for storing the matrix elements values.
    */
   using ConstValuesViewType = typename ValuesViewType::ConstViewType;

   /**
    * \brief Type of constant container view used for storing the column indexes of the matrix elements.
    */
   using ConstDiagonalsOffsetsView = typename DiagonalsOffsetsView::ConstViewType;

   /**
    * \brief Type of constant sparse matrix row view.
    */
   using RowView = MultidiagonalMatrixRowView< ValuesViewType, IndexerType, DiagonalsOffsetsView >;

   /**
    * \brief Type of constant sparse matrix row view.
    */
   using ConstRowView = MultidiagonalMatrixRowView< ConstValuesViewType, IndexerType, ConstDiagonalsOffsetsView >;

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
    * \param diagonalsOffsets is a vector view holding offsets of matrix diagonals,
    * \param values is a vector view holding values of matrix elements.
    * \param indexer is object responsible for indexing and organization of matrix elements
    */
   __cuda_callable__
   MultidiagonalMatrixRowView( IndexType rowIdx,
                               const DiagonalsOffsetsView& diagonalsOffsets,
                               const ValuesViewType& values,
                               const IndexerType& indexer );

   /**
    * \brief Returns number of diagonals of the multidiagonal matrix.
    *
    * \return number of diagonals of the multidiagonal matrix.
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

   DiagonalsOffsetsView diagonalsOffsets;

   ValuesViewType values;

   Indexer indexer;
};

}  // namespace TNL::Matrices

#include <TNL/Matrices/MultidiagonalMatrixRowView.hpp>

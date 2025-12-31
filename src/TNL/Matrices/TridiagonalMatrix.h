// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include "TridiagonalMatrixView.h"

namespace TNL::Matrices {

/**
 * \brief Implementation of sparse tridiagonal matrix.
 *
 * Use this matrix type for storing of tridiagonal matrices i.e., matrices having
 * non-zero matrix elements only on its diagonal and immediately above and bellow the diagonal.
 * This is an example:
 * \f[
 * \left(
 * \begin{array}{ccccccc}
 *  4  & -1  &  .  & .   &  . & .  \\
 * -1  &  4  & -1  &  .  &  . & .  \\
 *  .  & -1  &  4  & -1  &  . & .  \\
 *  .  &  .  & -1  &  4  & -1 &  . \\
 *  .  &  .  &  .  & -1  &  4 & -1 \\
 *  .  &  .  &  .  &  .  & -1 &  4
 * \end{array}
 * \right)
 * \f]
 *
 * Advantage is that we do not store the column indexes
 * explicitly as it is in \ref SparseMatrix. This can reduce significantly the
 * memory requirements which also means better performance. See the following table
 * for the storage requirements comparison between \ref TridiagonalMatrix and \ref SparseMatrix.
 *
 *  Real   | Index      |      SparseMatrix    | TridiagonalMatrix   | Ratio
 * --------|------------|----------------------|---------------------|-------
 *  float  | 32-bit int | 8 bytes per element  | 4 bytes per element | 50%
 *  double | 32-bit int | 12 bytes per element | 8 bytes per element | 75%
 *  float  | 64-bit int | 12 bytes per element | 4 bytes per element | 30%
 *  double | 64-bit int | 16 bytes per element | 8 bytes per element | 50%
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 * \tparam RealAllocator is allocator for the matrix elements.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class TridiagonalMatrix : public TridiagonalMatrixBase< Real, Device, Index, Organization >
{
   using Base = TridiagonalMatrixBase< Real, Device, Index, Organization >;

public:
   /**
    * \brief Type of vector holding values of matrix elements.
    */
   using ValuesVectorType = Containers::Vector< Real, Device, Index, RealAllocator >;

   /**
    * \brief The allocator for matrix elements values.
    */
   using RealAllocatorType = RealAllocator;

   /**
    * \brief Type of related matrix view.
    *
    * See \ref TridiagonalMatrixView.
    */
   using ViewType = TridiagonalMatrixView< Real, Device, Index, Organization >;

   /**
    * \brief Matrix view type for constant instances.
    *
    * See \ref TridiagonalMatrixView.
    */
   using ConstViewType = typename ViewType::ConstViewType;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             ElementsOrganization _Organization =
                Algorithms::Segments::DefaultElementsOrganization< _Device >::getOrganization(),
             typename _RealAllocator = typename Allocators::Default< _Device >::template Allocator< _Real > >
   using Self = TridiagonalMatrix< _Real, _Device, _Index, _Organization, _RealAllocator >;

   /**
    * \brief Type of related constant matrix.
    */
   using ConstMatrixType = TridiagonalMatrix< std::add_const_t< Real >, Device, Index, Organization, RealAllocator >;

   /**
    * \brief Constructor with no parameters.
    */
   TridiagonalMatrix() = default;

   /**
    * \brief Constructor with matrix dimensions.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    */
   TridiagonalMatrix( Index rows, Index columns );

   /**
    * \brief Constructor with matrix dimensions, diagonals offsets and matrix elements.
    *
    * The number of matrix rows is deduced from the size of the initializer list \e data.
    *
    * \tparam ListReal is type used in the initializer list defining matrix elements values.
    *
    * \param columns is number of matrix columns.
    * \param data is initializer list holding matrix elements. The size of the outer list
    *    defines the number of matrix rows. Each inner list defines values of each sub-diagonal
    *    and so its size should be lower or equal to three. Values
    *    of sub-diagonals which do not fit to given row are omitted.
    *
    * \par Example
    * \include Matrices/TridiagonalMatrix/TridiagonalMatrixExample_Constructor_init_list_1.cpp
    * \par Output
    * \include TridiagonalMatrixExample_Constructor_init_list_1.out
    */
   template< typename ListReal >
   TridiagonalMatrix( Index columns, const std::initializer_list< std::initializer_list< ListReal > >& data );

   /**
    * \brief Copy constructor.
    */
   explicit TridiagonalMatrix( const TridiagonalMatrix& matrix );

   /**
    * \brief Move constructor.
    */
   TridiagonalMatrix( TridiagonalMatrix&& ) noexcept = default;

   /**
    * \brief Returns a modifiable view of the tridiagonal matrix.
    *
    * See \ref TridiagonalMatrixView.
    *
    * \return tridiagonal matrix view.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the tridiagonal matrix.
    *
    * See \ref TridiagonalMatrixView.
    *
    * \return tridiagonal matrix view.
    */
   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Set matrix dimensions.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    */
   void
   setDimensions( Index rows, Index columns );

   /**
    * \brief Setup the matrix dimensions and diagonals offsets based on another tridiagonal matrix.
    *
    * \tparam Real_ is \e Real type of the source matrix.
    * \tparam Device_ is \e Device type of the source matrix.
    * \tparam Index_ is \e Index type of the source matrix.
    * \tparam Organization_ is \e Organization of the source matrix.
    * \tparam RealAllocator_ is \e RealAllocator of the source matrix.
    *
    * \param matrix is the source matrix.
    */
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
   void
   setLike( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix );

   /**
    * \brief This method is for compatibility with \ref SparseMatrix.
    *
    * It checks if the number of matrix diagonals is compatible with
    * required number of non-zero matrix elements in each row. If not
    * exception is thrown.
    *
    * \tparam RowCapacitiesVector is vector-like container type for holding required
    *    row capacities.
    *
    * \param rowCapacities is vector-like container holding required row capacities.
    */
   template< typename RowCapacitiesVector >
   void
   setRowCapacities( const RowCapacitiesVector& rowCapacities );

   /**
    * \brief Set matrix elements from an initializer list.
    *
    * \tparam ListReal is data type of the initializer list.
    *
    * \param data is initializer list holding matrix elements. The size of the outer list
    *    defines the number of matrix rows. Each inner list defines values of each sub-diagonal
    *    and so its size should be lower or equal to three. Values
    *    of sub-diagonals which do not fit to given row are omitted.
    *
    * \par Example
    * \include Matrices/TridiagonalMatrix/TridiagonalMatrixExample_setElements.cpp
    * \par Output
    * \include TridiagonalMatrixExample_setElements.out
    */
   template< typename ListReal >
   void
   setElements( const std::initializer_list< std::initializer_list< ListReal > >& data );

   /**
    * \brief Resets the matrix to zero dimensions.
    */
   void
   reset();

   // TODO: refactor this to a free function
   template< typename Real2, typename Index2 >
   void
   getTransposition( const TridiagonalMatrix< Real2, Device, Index2 >& matrix, const Real& matrixMultiplicator = 1.0 );

   /**
    * \brief Assignment of exactly the same matrix type.
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   TridiagonalMatrix&
   operator=( const TridiagonalMatrix& matrix );

   /**
    * \brief Assignment of another tridiagonal matrix
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
   TridiagonalMatrix&
   operator=( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix );

   /**
    * \brief Method for saving the matrix to the file with given filename.
    *
    * \param fileName is name of the file.
    */
   void
   save( const String& fileName ) const;

   /**
    * \brief Method for loading the matrix from the file with given filename.
    *
    * \param fileName is name of the file.
    */
   void
   load( const String& fileName );

protected:
   //! \brief Vector containing the allocated matrix elements.
   ValuesVectorType values;
};

/**
 * \brief Deserialization of tridiagonal matrices from binary files.
 */
template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
File&
operator>>( File& file, TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >& matrix );

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
File&
operator>>( File&& file, TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >& matrix );

}  // namespace TNL::Matrices

#include "TridiagonalMatrix.hpp"

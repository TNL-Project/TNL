// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Object.h>
#include <TNL/Containers/Vector.h>
#include "MultidiagonalMatrixView.h"

namespace TNL::Matrices {

/**
 * \brief Implementation of sparse multidiagonal matrix.
 *
 * Use this matrix type for storing of matrices where the offsets of non-zero elements
 * from the diagonal are the same in each row. Typically such matrices arise from
 * discretization of partial differential equations on regular numerical grids. This is
 * one example (dots represent zero matrix elements):
 *
 * \f[
 * \left(
 * \begin{array}{ccccccc}
 *  4  & -1  &  .  & -1  &  . & .  \\
 * -1  &  4  & -1  &  .  & -1 & .  \\
 *  .  & -1  &  4  & -1  &  . & -1 \\
 * -1  & .   & -1  &  4  & -1 &  . \\
 *  .  & -1  &  .  & -1  &  4 & -1 \\
 *  .  &  .  & -1  &  .  & -1 &  4
 * \end{array}
 * \right)
 * \f]
 *
 * In this matrix, the column indexes in each row \f$i\f$ can be expressed as
 * \f$\{i-3, i-1, i, i+1, i+3\}\f$ (where the resulting index is non-negative and
 *  smaller than the number of matrix columns). Therefore the diagonals offsets
 * are \f$\{-3,-1,0,1,3\}\f$. Advantage is that we do not store the column indexes
 * explicitly as it is in \ref SparseMatrix. This can reduce significantly the
 * memory requirements which also means better performance. See the following table
 * for the storage requirements comparison between \ref TNL::Matrices::MultidiagonalMatrix
 * and \ref TNL::Matrices::SparseMatrix.
 *
 *  Real   | Index     |      SparseMatrix    | MultidiagonalMatrix | Ratio
 * --------|-----------|----------------------|---------------------|-------
 *  float  | 32-bit int| 8 bytes per element  | 4 bytes per element | 50%
 *  double | 32-bit int| 12 bytes per element | 8 bytes per element | 75%
 *  float  | 64-bit int| 12 bytes per element | 4 bytes per element | 30%
 *  double | 64-bit int| 16 bytes per element | 8 bytes per element | 50%
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 * \tparam RealAllocator is allocator for the matrix elements.
 * \tparam IndexAllocator is allocator for the matrix elements offsets.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class MultidiagonalMatrix : public TNL::Object, public MultidiagonalMatrixBase< Real, Device, Index, Organization >
{
   using Base = MultidiagonalMatrixBase< Real, Device, Index, Organization >;

public:
   /**
    * \brief Type of vector holding values of matrix elements.
    */
   using ValuesVectorType = Containers::Vector< Real, Device, Index, RealAllocator >;

   // TODO: add documentation for these types
   using DiagonalOffsetsType = Containers::Vector< Index, Device, Index, IndexAllocator >;
   using HostDiagonalOffsetsType = Containers::Vector< Index, Devices::Host, Index >;

   /**
    * \brief The allocator for matrix elements values.
    */
   using RealAllocatorType = RealAllocator;

   /**
    * \brief The allocator for matrix elements offsets from the diagonal.
    */
   using IndexAllocatorType = IndexAllocator;

   /**
    * \brief Type of related matrix view.
    *
    * See \ref MultidiagonalMatrixView.
    */
   using ViewType = MultidiagonalMatrixView< Real, Device, Index, Organization >;

   /**
    * \brief Matrix view type for constant instances.
    *
    * See \ref MultidiagonalMatrixView.
    */
   using ConstViewType = MultidiagonalMatrixView< std::add_const_t< Real >, Device, Index, Organization >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             ElementsOrganization _Organization = Organization,
             typename _RealAllocator = typename Allocators::Default< _Device >::template Allocator< _Real >,
             typename _IndexAllocator = typename Allocators::Default< _Device >::template Allocator< _Index > >
   using Self = MultidiagonalMatrix< _Real, _Device, _Index, _Organization, _RealAllocator, _IndexAllocator >;

   /**
    * \brief Constructor with no parameters.
    */
   MultidiagonalMatrix() = default;

   /**
    * \brief Constructor with matrix dimensions.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    */
   MultidiagonalMatrix( Index rows, Index columns );

   /**
    * \brief Constructor with matrix dimensions and matrix elements offsets.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param diagonalOffsets are offsets of subdiagonals from the main diagonal.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor.cpp
    * \par Output
    * \include MultidiagonalMatrixExample_Constructor.out
    */
   template< typename Vector >
   MultidiagonalMatrix( Index rows, Index columns, const Vector& diagonalOffsets );

   /**
    * \brief Constructor with matrix dimensions and diagonals offsets.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param diagonalOffsets are offsets of sub-diagonals from the main diagonal.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor_init_list_1.cpp
    * \par Output
    * \include MultidiagonalMatrixExample_Constructor_init_list_1.out
    */
   template< typename ListIndex >
   MultidiagonalMatrix( Index rows, Index columns, std::initializer_list< ListIndex > diagonalOffsets );

   /**
    * \brief Constructor with matrix dimensions, diagonals offsets and matrix elements.
    *
    * The number of matrix rows is deduced from the size of the initializer list \e data.
    *
    * \tparam ListIndex is type used in the initializer list defining matrix diagonals offsets.
    * \tparam ListReal is type used in the initializer list defining matrix elements values.
    *
    * \param columns is number of matrix columns.
    * \param diagonalOffsets are offsets of sub-diagonals from the main diagonal.
    * \param data is initializer list holding matrix elements. The size of the outer list
    *    defines the number of matrix rows. Each inner list defines values of each sub-diagonal
    *    and so its size should be lower or equal to the size of \e diagonalOffsets. Values
    *    of sub-diagonals which do not fit to given row are omitted.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_Constructor_init_list_2.cpp
    * \par Output
    * \include MultidiagonalMatrixExample_Constructor_init_list_2.out
    */
   template< typename ListIndex, typename ListReal >
   MultidiagonalMatrix( Index columns,
                        std::initializer_list< ListIndex > diagonalOffsets,
                        const std::initializer_list< std::initializer_list< ListReal > >& data );

   /**
    * \brief Copy constructor.
    */
   MultidiagonalMatrix( const MultidiagonalMatrix& matrix );

   /**
    * \brief Move constructor.
    */
   MultidiagonalMatrix( MultidiagonalMatrix&& ) noexcept = default;

   /**
    * \brief Returns a modifiable view of the multidiagonal matrix.
    *
    * See \ref MultidiagonalMatrixView.
    *
    * \return multidiagonal matrix view.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the multidiagonal matrix.
    *
    * See \ref MultidiagonalMatrixView.
    *
    * \return multidiagonal matrix view.
    */
   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Set matrix dimensions and diagonals offsets.
    *
    * \tparam Vector is type of vector like container holding the diagonals offsets.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param diagonalOffsets is vector with diagonals offsets.
    */
   template< typename Vector >
   void
   setDimensions( Index rows, Index columns, const Vector& diagonalOffsets );

   /**
    * @brief Set the diagonals offsets by means of vector-like container.
    *
    * This method deletes current matrix elements.
    *
    * @tparam Vector is a type of vector-like container holding the subdiagonals offsets.
    * @param diagonalOffsets  is a vector-like container holding the subdiagonals offsets.
    */
   template< typename Vector >
   void
   setDiagonalOffsets( const Vector& diagonalOffsets );

   /**
    * @brief Set the diagonals offsets by means of initializer list.
    *
    * This method deletes current matrix elements.
    *
    * @tparam ListIndex is type of indexes used for the subdiagonals offsets definition.
    * @param diagonalOffsets is a initializer list with subdiagonals offsets.
    */
   template< typename ListIndex >
   void
   setDiagonalOffsets( std::initializer_list< ListIndex > diagonalOffsets );

   /**
    * \brief Setup the matrix dimensions and diagonals offsets based on another multidiagonal matrix.
    *
    * \tparam Real_ is \e Real type of the source matrix.
    * \tparam Device_ is \e Device type of the source matrix.
    * \tparam Index_ is \e Index type of the source matrix.
    * \tparam Organization_ is \e Organization of the source matrix.
    * \tparam RealAllocator_ is \e RealAllocator of the source matrix.
    * \tparam IndexAllocator_ is \e IndexAllocator of the source matrix.
    *
    * \param matrix is the source matrix.
    */
   template< typename Real_,
             typename Device_,
             typename Index_,
             ElementsOrganization Organization_,
             typename RealAllocator_,
             typename IndexAllocator_ >
   void
   setLike( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix );

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
    *    and so its size should be lower or equal to the size of \e diagonalsOffsets. Values
    *    of sub-diagonals which do not fit to given row are omitted.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixExample_setElements.cpp
    * \par Output
    * \include MultidiagonalMatrixExample_setElements.out
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
   getTransposition( const MultidiagonalMatrix< Real2, Device, Index2 >& matrix, const Real& matrixMultiplicator = 1.0 );

   /**
    * \brief Assignment of exactly the same matrix type.
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   MultidiagonalMatrix&
   operator=( const MultidiagonalMatrix& matrix );

   /**
    * \brief Assignment of another multidiagonal matrix
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   template< typename Real_,
             typename Device_,
             typename Index_,
             ElementsOrganization Organization_,
             typename RealAllocator_,
             typename IndexAllocator_ >
   MultidiagonalMatrix&
   operator=( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix );

   /**
    * \brief Method for saving the matrix to a file.
    *
    * \param file is the output file.
    */
   void
   save( File& file ) const override;

   /**
    * \brief Method for loading the matrix from a file.
    *
    * \param file is the input file.
    */
   void
   load( File& file ) override;

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

   // FIXME
   using Base::getSerializationType;

protected:
   //! \brief Vector containing the allocated matrix elements.
   ValuesVectorType values;

   // TODO: add documentation for these attributes
   DiagonalOffsetsType diagonalOffsets;
   HostDiagonalOffsetsType hostDiagonalOffsets;
};

}  // namespace TNL::Matrices

#include "MultidiagonalMatrix.hpp"

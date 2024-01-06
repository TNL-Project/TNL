// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>

#include <TNL/Object.h>
#include <TNL/Allocators/Default.h>

#include "DenseMatrixView.h"

namespace TNL::Matrices {

/**
 * \brief Implementation of dense matrix, i.e. matrix storing explicitly all of its elements including zeros.
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
class DenseMatrix : public TNL::Object, public DenseMatrixBase< Real, Device, Index, Organization >
{
   using Base = DenseMatrixBase< Real, Device, Index, Organization >;

public:
   /**
    * \brief The allocator for matrix elements.
    */
   using RealAllocatorType = RealAllocator;

   /**
    * \brief Type of related matrix view.
    *
    * See \ref DenseMatrixView.
    */
   using ViewType = DenseMatrixView< Real, Device, Index, Organization >;

   /**
    * \brief Matrix view type for constant instances.
    *
    * See \ref DenseMatrixView.
    */
   using ConstViewType = typename DenseMatrixView< Real, Device, Index, Organization >::ConstViewType;

   /**
    * \brief Type of vector holding values of matrix elements.
    */
   using ValuesVectorType = Containers::Vector< Real, Device, Index, RealAllocator >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             ElementsOrganization _Organization =
                Algorithms::Segments::DefaultElementsOrganization< _Device >::getOrganization(),
             typename _RealAllocator = typename Allocators::Default< _Device >::template Allocator< _Real > >
   using Self = DenseMatrix< _Real, _Device, _Index, _Organization, _RealAllocator >;

   /**
    * \brief Constructor only with values allocator.
    *
    * \param allocator is used for allocation of matrix elements values.
    */
   DenseMatrix( const RealAllocatorType& allocator = RealAllocatorType() );

   /**
    * \brief Copy constructor.
    *
    * \param matrix is the source matrix
    */
   DenseMatrix( const DenseMatrix& matrix );

   /**
    * \brief Move constructor.
    *
    * \param matrix is the source matrix
    */
   DenseMatrix( DenseMatrix&& matrix ) noexcept = default;

   /**
    * \brief Constructor with matrix dimensions.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param allocator is used for allocation of matrix elements values.
    */
   DenseMatrix( Index rows, Index columns, const RealAllocatorType& allocator = RealAllocatorType() );

   /**
    * \brief Constructor with 2D initializer list.
    *
    * The number of matrix rows is set to the outer list size and the number
    * of matrix columns is set to maximum size of inner lists. Missing elements
    * are filled in with zeros.
    *
    * \param data is a initializer list of initializer lists representing
    * list of matrix rows.
    * \param allocator is used for allocation of matrix elements values.
    *
    * \par Example
    * \include Matrices/DenseMatrix/DenseMatrixExample_Constructor_init_list.cpp
    * \par Output
    * \include DenseMatrixExample_Constructor_init_list.out
    */
   template< typename Value >
   DenseMatrix( std::initializer_list< std::initializer_list< Value > > data,
                const RealAllocatorType& allocator = RealAllocatorType() );

   /**
    * \brief Returns a modifiable view of the dense matrix.
    *
    * See \ref DenseMatrixView.
    *
    * \return dense matrix view.
    */
   [[nodiscard]] ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the dense matrix.
    *
    * See \ref DenseMatrixView.
    *
    * \return dense matrix view.
    */
   [[nodiscard]] ConstViewType
   getConstView() const;

   /**
    * \brief Set number of rows and columns of this matrix.
    *
    * \param rows is the number of matrix rows.
    * \param columns is the number of matrix columns.
    */
   void
   setDimensions( Index rows, Index columns );

   /**
    * \brief Set the number of matrix rows and columns by the given matrix.
    *
    * \tparam Matrix is matrix type. This can be any matrix having methods
    *  \ref getRows and \ref getColumns.
    *
    * \param matrix in the input matrix dimensions of which are to be adopted.
    */
   template< typename Matrix >
   void
   setLike( const Matrix& matrix );

   /**
    * \brief This method is only for the compatibility with the sparse matrices.
    *
    * This method does nothing. In debug mode it contains assertions checking
    * that given rowCapacities are compatible with the current matrix dimensions.
    */
   template< typename RowCapacitiesVector >
   void
   setRowCapacities( const RowCapacitiesVector& rowCapacities );

   /**
    * \brief This method recreates the dense matrix from 2D initializer list.
    *
    * The number of matrix rows is set to the outer list size and the number
    * of matrix columns is set to maximum size of inner lists. Missing elements
    * are filled in with zeros.
    *
    * \param data is a initializer list of initializer lists representing
    * list of matrix rows.
    *
    * \par Example
    * \include Matrices/DenseMatrix/DenseMatrixExample_setElements.cpp
    * \par Output
    * \include DenseMatrixExample_setElements.out
    */
   template< typename Value >
   void
   setElements( std::initializer_list< std::initializer_list< Value > > data );

   /**
    * \brief This method sets the dense matrix elements from std::map.
    *
    * This is intended for compatibility with \ref SparseMatrix, the method
    * is used e.g. in \ref MatrixReader.
    *
    * The matrix elements values are given as a map \e data where keys are
    * std::pair of matrix coordinates ( {row, column} ) and value is the
    * matrix element value.
    *
    * \tparam MapIndex is a type for indexing rows and columns.
    * \tparam MapValue is a type for matrix elements values in the map.
    *
    * \param map is std::map containing matrix elements.
    */
   template< typename MapIndex, typename MapValue >
   void
   setElements( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map );

   /**
    * \brief Resets the matrix to zero dimensions.
    */
   void
   reset();

   // TODO: refactor this to a free function
   template< typename Matrix1, typename Matrix2, int tileDim = 32 >
   void
   getMatrixProduct( const Matrix1& matrix1, const Matrix2& matrix2, Real matrixMultiplicator = 1.0 );

   // TODO: refactor this to a free function
   template< typename Matrix, int tileDim = 32 >
   void
   getTransposition( const Matrix& matrix, Real matrixMultiplicator = 1.0 );

   /**
    * \brief Copy-assignment of exactly the same matrix type.
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   DenseMatrix&
   operator=( const DenseMatrix& matrix );

   /**
    * \brief Move-assignment of exactly the same matrix type.
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   DenseMatrix&
   operator=( DenseMatrix&& matrix ) noexcept( false );

   /**
    * \brief Assignment operator with the same organization.
    *
    * \param matrix is the right-hand side matrix.
    * \return reference to this matrix.
    */
   template< typename RHSReal, typename RHSDevice, typename RHSIndex, typename RHSRealAllocator >
   DenseMatrix&
   operator=( const DenseMatrix< RHSReal, RHSDevice, RHSIndex, Organization, RHSRealAllocator >& matrix );

   /**
    * \brief Assignment operator with matrix view having the same elements organization.
    *
    * \param matrix is the right-hand side matrix.
    * \return reference to this matrix.
    */
   template< typename RHSReal, typename RHSDevice, typename RHSIndex >
   DenseMatrix&
   operator=( const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, Organization >& matrix );

   /**
    * \brief Assignment operator with other dense matrices.
    *
    * \param matrix is the right-hand side matrix.
    * \return reference to this matrix.
    */
   template< typename RHSReal,
             typename RHSDevice,
             typename RHSIndex,
             ElementsOrganization RHSOrganization,
             typename RHSRealAllocator >
   DenseMatrix&
   operator=( const DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSOrganization, RHSRealAllocator >& matrix );

   /**
    * \brief Assignment operator with other dense matrices.
    *
    * \param matrix is the right-hand side matrix.
    * \return reference to this matrix.
    */
   template< typename RHSReal, typename RHSDevice, typename RHSIndex, ElementsOrganization RHSOrganization >
   DenseMatrix&
   operator=( const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, RHSOrganization >& matrix );

   /**
    * \brief Assignment operator with other (sparse) types of matrices.
    *
    * \param matrix is the right-hand side matrix.
    * \return reference to this matrix.
    */
   template< typename RHSMatrix >
   DenseMatrix&
   operator=( const RHSMatrix& matrix );

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

   /**
    * \brief Method for saving the matrix to a file.
    *
    * \param file is the file where the matrix will be saved.
    */
   void
   save( File& file ) const override;

   /**
    * \brief Method for loading the matrix from a file.
    *
    * \param file is the file from which the matrix will be loaded.
    */
   void
   load( File& file ) override;

   // FIXME
   using Base::getSerializationType;

protected:
   //! \brief Vector containing the allocated matrix elements.
   ValuesVectorType values;

   //! \brief Instance of the segments used for indexing in the dense matrix.
   typename Base::SegmentsType segments;
};

}  // namespace TNL::Matrices

#include "DenseMatrix.hpp"

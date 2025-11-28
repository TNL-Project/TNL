// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <type_traits>

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

/**
 * \brief Namespace for matrix formats.
 */
namespace TNL::Matrices {

using Algorithms::Segments::ElementsOrganization;

/**
 * \brief Padding index value.
 *
 * Padding index is used for column indexes of padding zeros. Padding zeros
 * are used in some sparse matrix formats for better data alignment in memory.
 */
template< typename Index >
constexpr Index paddingIndex = static_cast< Index >( -1 );

/**
 * \brief Encoding of the matrix elements in initializer lists
 * or STL maps.
 */
enum class MatrixElementsEncoding : std::uint8_t
{
   Complete,        //!<  All elements of the matrix are provided.
   SymmetricLower,  //!<  Only lower part of the matrix is provided.
   SymmetricUpper,  //!<  Only upper part of the matrix is provided.
   SymmetricMixed   //!<  For each couple of non-zero elements a_ij and a_ji, at least one is provided. It is handy for example
                    //!<  for adjacency matrices of undirected graphs.
};

/**
 * \brief Base class for the implementation of concrete matrix types.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 */
template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
class MatrixBase
{
public:
   /**
    * \brief The type of matrix elements.
    */
   using RealType = std::remove_const_t< Real >;

   /**
    * \brief The device where the matrix is allocated.
    */
   using DeviceType = Device;

   /**
    * \brief The type used for matrix elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Type of vector view holding values of matrix elements.
    */
   using ValuesViewType = Containers::VectorView< Real, Device, Index >;

   /**
    * \brief Type of constant vector view holding values of matrix elements.
    */
   using ConstValuesViewType = typename ValuesViewType::ConstViewType;

   // TODO: add documentation for this type (it is also questionable if it should be in MatrixBase or SparseMatrixBase)
   using RowCapacitiesType = Containers::Vector< Index, Device, Index >;

   /**
    * \brief Matrix elements organization getter.
    *
    * \return matrix elements organization - RowMajorOrder of ColumnMajorOrder.
    */
   [[nodiscard]] static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }

   /**
    * \brief Test of binary matrix type.
    *
    * \return \e true if the matrix is stored as binary and \e false otherwise.
    */
   [[nodiscard]] static constexpr bool
   isBinary()
   {
      return std::is_same_v< std::decay_t< Real >, bool >;
   }

   /**
    * \brief Test of symmetric matrix type.
    *
    * \return \e true if the matrix is stored as symmetric and \e false otherwise.
    */
   [[nodiscard]] static constexpr bool
   isSymmetric()
   {
      return MatrixType::isSymmetric();
   }

   /**
    * \brief Basic constructor with no parameters.
    */
   __cuda_callable__
   MatrixBase() = default;

   /**
    * \brief Constructor with matrix dimensions and matrix elements values.
    *
    * The matrix elements values are passed in a form vector view.
    *
    * \param rows is a number of matrix rows.
    * \param columns is a number of matrix columns.
    * \param values is a vector view with matrix elements values.
    */
   __cuda_callable__
   MatrixBase( IndexType rows, IndexType columns, ValuesViewType values );

   /**
    * \brief Shallow copy constructor.
    *
    * \param view is an input matrix view.
    */
   __cuda_callable__
   MatrixBase( const MatrixBase& view ) = default;

   /**
    * \brief Move constructor.
    *
    * \param view is an input matrix view.
    */
   __cuda_callable__
   MatrixBase( MatrixBase&& view ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because matrix assignment in general requires
    * reallocation.
    */
   __cuda_callable__
   MatrixBase&
   operator=( const MatrixBase& ) = delete;

   /**
    * \brief Move-assignment operator.
    */
   __cuda_callable__
   MatrixBase&
   operator=( MatrixBase&& ) = delete;

   /**
    * \brief Tells the number of allocated matrix elements.
    *
    * In the case of dense matrices, this is just product of the number of rows and the number of columns.
    * But for other matrix types like sparse matrices, this can be different.
    *
    * \return Number of allocated matrix elements.
    */
   [[nodiscard]] IndexType
   getAllocatedElementsCount() const;

   /**
    * \brief Returns number of matrix rows.
    *
    * \return number of matrix row.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getRows() const;

   /**
    * \brief Returns number of matrix columns.
    *
    * \return number of matrix columns.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getColumns() const;

   /**
    * \brief Returns a constant reference to a vector with the matrix elements values.
    *
    * \return constant reference to a vector with the matrix elements values.
    */
   [[nodiscard]] __cuda_callable__
   const ValuesViewType&
   getValues() const;

   /**
    * \brief Returns a reference to a vector with the matrix elements values.
    *
    * \return constant reference to a vector with the matrix elements values.
    */
   [[nodiscard]] __cuda_callable__
   ValuesViewType&
   getValues();

protected:
   IndexType rows = 0;
   IndexType columns = 0;

   ValuesViewType values;

   /**
    * \brief Re-initializes the internal attributes of the base class.
    *
    * Note that this function is \e protected to ensure that the user cannot
    * modify the base class of a matrix. For the same reason, in future code
    * development we also need to make sure that all non-const functions in
    * the base class return by value and not by reference.
    */
   __cuda_callable__
   void
   bind( IndexType rows, IndexType columns, ValuesViewType values );
};

}  // namespace TNL::Matrices

#include "MatrixBase.hpp"

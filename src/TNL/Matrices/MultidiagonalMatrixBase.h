// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "MatrixBase.h"
#include "MultidiagonalMatrixRowView.h"
#include "details/MultidiagonalMatrixIndexer.h"

namespace TNL::Matrices {

/**
 * \brief A common base class for \ref MultidiagonalMatrix and \ref MultidiagonalMatrixView.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 */
template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
class MultidiagonalMatrixBase : public MatrixBase< Real, Device, Index, GeneralMatrix, Organization >
{
   using Base = MatrixBase< Real, Device, Index, GeneralMatrix, Organization >;

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

   // TODO: add documentation for these types
   using IndexerType = details::MultidiagonalMatrixIndexer< Index, Organization == Algorithms::Segments::RowMajorOrder >;
   using DiagonalOffsetsView = Containers::
      VectorView< std::conditional_t< std::is_const< Real >::value, std::add_const_t< Index >, Index >, Device, Index >;
   using HostDiagonalOffsetsView = Containers::
      VectorView< std::conditional_t< std::is_const< Real >::value, std::add_const_t< Index >, Index >, Devices::Host, Index >;

   /**
    * \brief Type for accessing matrix rows.
    */
   using RowView = MultidiagonalMatrixRowView< typename Base::ValuesViewType, IndexerType, DiagonalOffsetsView >;

   /**
    * \brief Type for accessing constant matrix rows.
    */
   using ConstRowView = typename RowView::ConstRowView;

   /**
    * \brief Constructor with no parameters.
    */
   __cuda_callable__
   MultidiagonalMatrixBase() = default;

   /**
    * \brief Constructor with all necessary data and views.
    *
    * \param values is a vector view with matrix elements values
    * \param diagonalOffsets is a vector view with diagonals offsets
    * \param hostDiagonalOffsets is a vector view with a copy of diagonals offsets on the host
    * \param indexer is an indexer of matrix elements
    */
   __cuda_callable__
   MultidiagonalMatrixBase( typename Base::ValuesViewType values,
                            DiagonalOffsetsView diagonalOffsets,
                            HostDiagonalOffsetsView hostDiagonalOffsets,
                            IndexerType indexer );

   /**
    * \brief Copy constructor.
    */
   __cuda_callable__
   MultidiagonalMatrixBase( const MultidiagonalMatrixBase& ) = default;

   /**
    * \brief Move constructor.
    */
   __cuda_callable__
   MultidiagonalMatrixBase( MultidiagonalMatrixBase&& ) noexcept = default;

   /**
    * \brief Copy-assignment operator.
    *
    * It is a deleted function, because matrix assignment in general requires
    * reallocation.
    */
   MultidiagonalMatrixBase&
   operator=( const MultidiagonalMatrixBase& ) = delete;

   /**
    * \brief Move-assignment operator.
    */
   MultidiagonalMatrixBase&
   operator=( MultidiagonalMatrixBase&& ) = delete;

   /**
    * \brief Returns string with serialization type.
    *
    * The string has a form `Matrices::MultidiagonalMatrix< RealType,  [any_device], IndexType, Organization, [any_allocator],
    * [any_allocator] >`.
    *
    * See \ref MultidiagonalMatrix::getSerializationType.
    *
    * \return \ref String with the serialization type.
    */
   [[nodiscard]] static std::string
   getSerializationType();

   /**
    * \brief Returns number of diagonals.
    *
    * \return Number of diagonals.
    */
   [[nodiscard]] __cuda_callable__
   IndexType
   getDiagonalsCount() const;

   /**
    * \brief Compute capacities of all rows.
    *
    * The row capacities are not stored explicitly and must be computed.
    *
    * \param rowCapacities is a vector where the row capacities will be stored.
    */
   template< typename Vector >
   void
   getRowCapacities( Vector& rowCapacities ) const;

   /**
    * \brief Computes number of non-zeros in each row.
    *
    * \param rowLengths is a vector into which the number of non-zeros in each row
    * will be stored.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getCompressedRowLengths.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_getCompressedRowLengths.out
    */
   template< typename Vector >
   void
   getCompressedRowLengths( Vector& rowLengths ) const;

   /**
    * \brief Returns number of non-zero matrix elements.
    *
    * This method really counts the non-zero matrix elements and so
    * it returns zero for matrix having all allocated elements set to zero.
    *
    * \return number of non-zero matrix elements.
    */
   [[nodiscard]] IndexType
   getNonzeroElementsCount() const override;

   /**
    * \brief Comparison operator with another multidiagonal matrix.
    *
    * \tparam Real_ is \e Real type of the source matrix.
    * \tparam Device_ is \e Device type of the source matrix.
    * \tparam Index_ is \e Index type of the source matrix.
    * \tparam Organization_ is \e Organization of the source matrix.
    *
    * \param matrix is the source matrix view.
    *
    * \return \e true if both matrices are identical and \e false otherwise.
    */
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
   [[nodiscard]] bool
   operator==( const MultidiagonalMatrixBase< Real_, Device_, Index_, Organization_ >& matrix ) const;

   /**
    * \brief Comparison operator with another multidiagonal matrix.
    *
    * \tparam Real_ is \e Real type of the source matrix.
    * \tparam Device_ is \e Device type of the source matrix.
    * \tparam Index_ is \e Index type of the source matrix.
    * \tparam Organization_ is \e Organization of the source matrix.
    *
    * \param matrix is the source matrix view.
    *
    * \return \e true if both matrices are NOT identical and \e false otherwise.
    */
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
   [[nodiscard]] bool
   operator!=( const MultidiagonalMatrixBase< Real_, Device_, Index_, Organization_ >& matrix ) const;

   /**
    * \brief Non-constant getter of simple structure for accessing given matrix row.
    *
    * \param rowIdx is matrix row index.
    *
    * \return RowView for accessing given matrix row.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getRow.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_getRow.out
    *
    * See \ref MultidiagonalMatrixRowView.
    */
   [[nodiscard]] __cuda_callable__
   RowView
   getRow( IndexType rowIdx );

   /**
    * \brief Constant getter of simple structure for accessing given matrix row.
    *
    * \param rowIdx is matrix row index.
    *
    * \return RowView for accessing given matrix row.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getConstRow.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_getConstRow.out
    *
    * See \ref MultidiagonalMatrixRowView.
    */
   [[nodiscard]] __cuda_callable__
   ConstRowView
   getRow( IndexType rowIdx ) const;

   /**
    * \brief Set all matrix elements to given value.
    *
    * \param value is the new value of all matrix elements.
    */
   void
   setValue( const RealType& value );

   /**
    * \brief Sets element at given \e row and \e column to given \e value.
    *
    * This method can be called from the host system (CPU) no matter
    * where the matrix is allocated. If the matrix is allocated on GPU this method
    * can be called even from device kernels. If the matrix is allocated in GPU device
    * this method is called from CPU, it transfers values of each matrix element separately and so the
    * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
    * or \ref MultidiagonalMatrix::forElements and \ref MultidiagonalMatrix::forAllElements.
    * The call may fail if the matrix row capacity is exhausted.
    *
    * \param row is row index of the element.
    * \param column is columns index of the element.
    * \param value is the value the element will be set to.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_setElement.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_setElement.out
    */
   __cuda_callable__
   void
   setElement( IndexType row, IndexType column, const RealType& value );

   /**
    * \brief Add element at given \e row and \e column to given \e value.
    *
    * This method can be called from the host system (CPU) no matter
    * where the matrix is allocated. If the matrix is allocated on GPU this method
    * can be called even from device kernels. If the matrix is allocated in GPU device
    * this method is called from CPU, it transfers values of each matrix element separately and so the
    * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
    * or \ref MultidiagonalMatrix::forElements and \ref MultidiagonalMatrix::forAllElements.
    * The call may fail if the matrix row capacity is exhausted.
    *
    * \param row is row index of the element.
    * \param column is columns index of the element.
    * \param value is the value the element will be set to.
    * \param thisElementMultiplicator is multiplicator the original matrix element
    *   value is multiplied by before addition of given \e value.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_addElement.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_addElement.out
    */
   __cuda_callable__
   void
   addElement( IndexType row, IndexType column, const RealType& value, const RealType& thisElementMultiplicator = 1.0 );

   /**
    * \brief Returns value of matrix element at position given by its row and column index.
    *
    * This method can be called from the host system (CPU) no matter
    * where the matrix is allocated. If the matrix is allocated on GPU this method
    * can be called even from device kernels. If the matrix is allocated in GPU device
    * this method is called from CPU, it transfers values of each matrix element separately and so the
    * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
    * or \ref MultidiagonalMatrix::forElements and \ref MultidiagonalMatrix::forAllElements.
    *
    * \param row is a row index of the matrix element.
    * \param column i a column index of the matrix element.
    *
    * \return value of given matrix element.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getElement.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_getElement.out
    */
   [[nodiscard]] __cuda_callable__
   RealType
   getElement( IndexType row, IndexType column ) const;

   /**
    * \brief Method for performing general reduction on matrix rows for constant instances.
    *
    * \tparam Fetch is a type of lambda function for data fetch declared as
    *
    * ```
    * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue { ...
    * };
    * ```
    *
    *  The return type of this lambda can be any non void.
    * \tparam Reduce is a type of lambda function for reduction declared as
    *
    * ```
    * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
    * ```
    *
    * \tparam Keep is a type of lambda function for storing results of reduction in each row. It is declared as
    *
    * ```
    * auto keep = [=] __cuda_callable__ ( IndexType rowIdx, const RealType& value ) { ... };
    * ```
    *
    * \tparam FetchValue is type returned by the Fetch lambda function.
    *
    * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param fetch is an instance of lambda function for data fetch.
    * \param reduce is an instance of lambda function for reduction.
    * \param keep in an instance of lambda function for storing results.
    * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation, i.e. element which does not
    *                 change the result of the reduction.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_reduceRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_reduceRows.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceRows( IndexType begin, IndexType end, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const;

   /**
    * \brief Method for performing general reduction on all matrix rows for constant instances.
    *
    * \tparam Fetch is a type of lambda function for data fetch declared as
    *
    * ```
    * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue { ...
    * };
    * ```
    *
    * The return type of this lambda can be any non void.
    * \tparam Reduce is a type of lambda function for reduction declared as
    *
    * ```
    * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
    * ```
    *
    * \tparam Keep is a type of lambda function for storing results of reduction in each row. It is declared as
    *
    * ```
    * auto keep = [=] __cuda_callable__ ( IndexType rowIdx, const RealType& value ) { ... };
    * ```
    *
    * \tparam FetchValue is type returned by the Fetch lambda function.
    *
    * \param fetch is an instance of lambda function for data fetch.
    * \param reduce is an instance of lambda function for reduction.
    * \param keep in an instance of lambda function for storing results.
    * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation, i.e. element which does not
    *                 change the result of the reduction.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_reduceAllRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_reduceAllRows.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const;

   /**
    * \brief Method for iteration over all matrix rows for constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
    *
    * ```
    * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value )
    * { ... };
    * ```
    *
    * where
    *
    * \e rowIdx is an index of the matrix row.
    *
    * \e localIdx parameter is a rank of the non-zero element in given row. It is also, in fact,
    *  index of the matrix subdiagonal.
    *
    * \e columnIdx is a column index of the matrix element.
    *
    * \e value is the matrix element value.
    *
    * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called in each row.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forRows.out
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function& function ) const;

   /**
    * \brief Method for iteration over all matrix rows for non-constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
    *
    * ```
    * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value )
    * { ... };
    * ```
    *
    * where
    *
    * \e rowIdx is an index of the matrix row.
    *
    * \e localIdx parameter is a rank of the non-zero element in given row. It is also, in fact,
    *  index of the matrix subdiagonal.
    *
    * \e columnIdx is a column index of the matrix element.
    *
    * \e value is a reference to the matrix element value. It can be used even for changing the matrix element value.
    *
    * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called in each row.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forRows.out
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function& function );

   /**
    * \brief This method calls \e forElements for all matrix rows (for constant instances).
    *
    * See \ref MultidiagonalMatrix::forElements.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called in each row.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forAllElements.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forAllElements.out
    */
   template< typename Function >
   void
   forAllElements( Function& function ) const;

   /**
    * \brief This method calls \e forElements for all matrix rows.
    *
    * See \ref MultidiagonalMatrix::forElements.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called in each row.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forAllElements.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forAllElements.out
    */
   template< typename Function >
   void
   forAllElements( Function& function );

   /**
    * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end).
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref MultidiagonalMatrixBase::forElements where more than one thread can be mapped to each row.
    *
    * \tparam Function is type of the lambda function.
    *
    * \param begin defines beginning of the range [ \e begin,\e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called for each row.
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::MultidiagonalMatrixBase::RowView.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forRows.out
    */
   template< typename Function >
   void
   forRows( IndexType begin, IndexType end, Function&& function );

   /**
    * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end) for constant instances.
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref MultidiagonalMatrixBase::forElements where more than one thread can be mapped to each row.
    *
    * \tparam Function is type of the lambda function.
    *
    * \param begin defines beginning of the range [ \e begin,\e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called for each row.
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::MultidiagonalMatrixBase::RowView.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forRows.out
    */
   template< typename Function >
   void
   forRows( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Method for parallel iteration over all matrix rows.
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref MultidiagonalMatrixBase::forAllElements where more than one thread can be mapped to each row.
    *
    * \tparam Function is type of the lambda function.
    *
    * \param function is an instance of the lambda function to be called for each row.
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::MultidiagonalMatrixBase::RowView.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forRows.out
    */
   template< typename Function >
   void
   forAllRows( Function&& function );

   /**
    * \brief Method for parallel iteration over all matrix rows for constant instances.
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref MultidiagonalMatrixBase::forAllElements where more than one thread can be mapped to each row.
    *
    * \tparam Function is type of the lambda function.
    *
    * \param function is an instance of the lambda function to be called for each row.
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::MultidiagonalMatrixBase::RowView.
    *
    * \par Example
    * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
    * \par Output
    * \include MultidiagonalMatrixViewExample_forRows.out
    */
   template< typename Function >
   void
   forAllRows( Function&& function ) const;

   /**
    * \brief Method for sequential iteration over all matrix rows for constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
    *
    * ```
    * auto function = [] __cuda_callable__ ( const RowView& row ) { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::MultidiagonalMatrixBase::RowView.
    *
    * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForRows( IndexType begin, IndexType end, Function& function ) const;

   /**
    * \brief Method for sequential iteration over all matrix rows for non-constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::MultidiagonalMatrixBase::RowView.
    *
    * \param begin defines beginning of the range [ \e  begin, \e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForRows( IndexType begin, IndexType end, Function& function );

   /**
    * \brief This method calls \e sequentialForRows for all matrix rows (for constant instances).
    *
    * See \ref MultidiagonalMatrixBase::sequentialForRows.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForAllRows( Function& function ) const;

   /**
    * \brief This method calls \e sequentialForRows for all matrix rows.
    *
    * See \ref MultidiagonalMatrixBase::sequentialForAllRows.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForAllRows( Function& function );

   /**
    * \brief Computes product of matrix and vector.
    *
    * More precisely, it computes:
    *
    * ```
    * outVector = matrixMultiplicator * ( * this ) * inVector + outVectorMultiplicator * outVector
    * ```
    *
    * \tparam InVector is type of input vector. It can be
    *         \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView,
    *         \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
    *         or similar container.
    * \tparam OutVector is type of output vector. It can be
    *         \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView,
    *         \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
    *         or similar container.
    *
    * \param inVector is input vector.
    * \param outVector is output vector.
    * \param matrixMultiplicator is a factor by which the matrix is multiplied. It is one by default.
    * \param outVectorMultiplicator is a factor by which the outVector is multiplied before added
    *    to the result of matrix-vector product. It is zero by default.
    * \param begin is the beginning of the rows range for which the vector product
    *    is computed. It is zero by default.
    * \param end is the end of the rows range for which the vector product
    *    is computed. It is number if the matrix rows by default.
    */
   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector,
                  OutVector& outVector,
                  RealType matrixMultiplicator = 1.0,
                  RealType outVectorMultiplicator = 0.0,
                  IndexType begin = 0,
                  IndexType end = 0 ) const;

   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
   void
   addMatrix( const MultidiagonalMatrixBase< Real_, Device_, Index_, Organization_ >& matrix,
              const RealType& matrixMultiplicator = 1.0,
              const RealType& thisMatrixMultiplicator = 1.0 );

   /**
    * \brief Method for printing the matrix to output stream.
    *
    * \param str is the output stream.
    */
   void
   print( std::ostream& str ) const;

   /**
    * \brief This method returns matrix elements indexer used by this matrix.
    *
    * \return constant reference to the indexer.
    */
   [[nodiscard]] __cuda_callable__
   const IndexerType&
   getIndexer() const;

   /**
    * \brief This method returns matrix elements indexer used by this matrix.
    *
    * \return non-constant reference to the indexer.
    */
   [[nodiscard]] __cuda_callable__
   IndexerType&
   getIndexer();

   /**
    * \brief Returns a view for the vector of diagonal offsets.
    */
   __cuda_callable__
   DiagonalOffsetsView
   getDiagonalOffsets();

   /**
    * \brief Returns a view for the vector of diagonal offsets.
    */
   __cuda_callable__
   typename DiagonalOffsetsView::ConstViewType
   getDiagonalOffsets() const;

protected:
   DiagonalOffsetsView diagonalOffsets;

   HostDiagonalOffsetsView hostDiagonalOffsets;

   IndexerType indexer;

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
   bind( typename Base::ValuesViewType values,
         DiagonalOffsetsView diagonalOffsets,
         HostDiagonalOffsetsView hostDiagonalOffsets,
         IndexerType indexer );
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
template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::ostream&
operator<<( std::ostream& str, const MultidiagonalMatrixBase< Real, Device, Index, Organization >& matrix )
{
   matrix.print( str );
   return str;
}

}  // namespace TNL::Matrices

#include "MultidiagonalMatrixBase.hpp"

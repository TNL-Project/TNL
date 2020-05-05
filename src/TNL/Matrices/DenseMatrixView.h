/***************************************************************************
                          DenseMatrixView.h  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrixRowView.h>
#include <TNL/Matrices/MatrixView.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Containers/Segments/Ellpack.h>

namespace TNL {
namespace Matrices {

/**
 * \brief Implementation of dense matrix view.
 * 
 * It serves as an accessor to \ref DenseMatrix for example when passing the
 * matrix to lambda functions. DenseMatrix view can be also created in CUDA kernels.
 * 
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam MatrixElementsOrganization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 * 
 * See \ref DenseMatrix.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Containers::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class DenseMatrixView : public MatrixView< Real, Device, Index >
{
   protected:
      using BaseType = Matrix< Real, Device, Index >;
      using ValuesVectorType = typename BaseType::ValuesVectorType;
      using SegmentsType = Containers::Segments::Ellpack< Device, Index, typename Allocators::Default< Device >::template Allocator< Index >, Organization, 1 >;
      using SegmentsViewType = typename SegmentsType::ViewType;
      using SegmentViewType = typename SegmentsType::SegmentViewType;

   public:

      /**
       * \brief The type of matrix elements.
       */
      using RealType = Real;

      /**
       * \brief The device where the matrix is allocated.
       */
      using DeviceType = Device;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = Index;

      /**
       * \brief Matrix elements organization getter.
       * 
       * \return matrix elements organization - RowMajorOrder of ColumnMajorOrder.
       */
      static constexpr ElementsOrganization getOrganization() { return Organization; };

      /**
       * \brief Matrix elements container view type.
       * 
       * Use this for embedding of the matrix elements values.
       */
      using ValuesViewType = typename ValuesVectorType::ViewType;

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
      using ConstViewType = DenseMatrixView< typename std::add_const< Real >::type, Device, Index, Organization >;

      /**
       * \brief Type for accessing matrix row.
       */
      using RowView = DenseMatrixRowView< SegmentViewType, ValuesViewType >;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index >
      using Self = DenseMatrixView< _Real, _Device, _Index >;

      /**
       * \brief Constructor without parameters.
       */
      __cuda_callable__
      DenseMatrixView();

      /**
       * \brief Constructor with matrix dimensions and values.
       * 
       * Organization of matrix elements values in 
       * 
       * \param rows number of matrix rows.
       * \param columns number of matrix columns.
       * \param values is vector view with matrix elements values.
       */
      __cuda_callable__
      DenseMatrixView( const IndexType rows,
                       const IndexType columns,
                       const ValuesViewType& values );

      /**
       * \brief Copy constructor.
       * 
       * \param matrix is the source matrix view.
       */
      __cuda_callable__
      DenseMatrixView( const DenseMatrixView& matrix ) = default;

      /**
       * \brief Returns a modifiable dense matrix view.
       * 
       * \return dense matrix view.
       */
      __cuda_callable__
      ViewType getView();

      /**
       * \brief Returns a non-modifiable dense matrix view.
       * 
       * \return dense matrix view.
       */
      __cuda_callable__
      ConstViewType getConstView() const;

      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      IndexType getMaxRowLength() const;

      IndexType getElementsCount() const;

      IndexType getNonzeroElementsCount() const;

      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );


      void setValue( const RealType& v );

      __cuda_callable__
      Real& operator()( const IndexType row,
                        const IndexType column );

      __cuda_callable__
      const Real& operator()( const IndexType row,
                              const IndexType column ) const;

      __cuda_callable__
      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      __cuda_callable__
      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      __cuda_callable__
      Real getElement( const IndexType row,
                       const IndexType column ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function ) const;

      template< typename Function >
      void forRows( IndexType first, IndexType last, Function& function );

      template< typename Function >
      void forAllRows( Function& function ) const;

      template< typename Function >
      void forAllRows( Function& function );

      template< typename InVector, typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType& matrixMultiplicator = 1.0,
                          const RealType& outVectorMultiplicator = 0.0,
                          const IndexType begin = 0,
                          IndexType end = 0 ) const;

      template< typename Matrix >
      void addMatrix( const Matrix& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename MatrixView1, typename MatrixView2, int tileDim = 32 >
      void getMatrixProduct( const MatrixView1& matrix1,
                             const MatrixView2& matrix2,
                             const RealType& matrix1Multiplicator = 1.0,
                             const RealType& matrix2Multiplicator = 1.0 );

      template< typename Matrix, int tileDim = 32 >
      void getTransposition( const Matrix& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      template< typename Vector1, typename Vector2 >
      void performSORIteration( const Vector1& b,
                                const IndexType row,
                                Vector2& x,
                                const RealType& omega = 1.0 ) const;

      DenseMatrixView& operator=( const DenseMatrixView& matrix );

      void save( const String& fileName ) const;

      void save( File& file ) const;

      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType column ) const;

      //typedef DenseDeviceDependentCode< DeviceType > DeviceDependentCode;
      //friend class DenseDeviceDependentCode< DeviceType >;

      SegmentsViewType segments;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/DenseMatrixView.hpp>

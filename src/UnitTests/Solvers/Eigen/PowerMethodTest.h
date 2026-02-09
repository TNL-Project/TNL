#include <tuple>
#if defined( __CUDACC__ )
   #include <TNL/Devices/Cuda.h>
#endif
#include <TNL/Devices/Host.h>
#include <TNL/Math.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Solvers/Eigen/experimental/PowerIteration.h>
#include <TNL/Solvers/Eigen/experimental/ShiftedPowerIteration.h>

#include <gtest/gtest.h>
#include <stdexcept>

template< typename RealType, typename Device >
void
checkPowerIterationDense0D()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   const MatrixTypeCMO A = {};
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = {};
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const MatrixTypeCMO B;

   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( B, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO C = {};
   VectorType initialVecRMO = {};
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( C, epsilon, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const MatrixTypeRMO D;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( D, epsilon, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationExceptionSizeSquare()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   const MatrixTypeCMO A = { { 1, 2, 3 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Power iteration is possible only for square matrices", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO B = { { 1, 2, 3 } };
   VectorType initialVecRMO = { 1, 1, 1 };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( B, epsilon, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Power iteration is possible only for square matrices", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationExceptionSizeVector()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   const MatrixTypeCMO A = { { 1 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must have the same size as the matrix", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO B = { { 1 } };
   VectorType initialVecRMO = { 1, 1 };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( B, epsilon, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must have the same size as the matrix", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationExceptionZeroVector()
{
   const TNL::Algorithms::Segments::ElementsOrganization organization = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixType = TNL::Matrices::DenseMatrix< RealType, Device, int, organization >;
   const MatrixType A = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 0, 0 };
   RealType epsilon = 1e-8;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( A, epsilon, initialVec, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must be nonzero", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO B = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   VectorType initialVecRMO = { 0, 0 };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( B, epsilon, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must be nonzero", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationDense1D()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   MatrixTypeCMO A = { { 1.0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 2.0 };
   RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVec, 100 );
   EXPECT_EQ( eigenvalue, 1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   A.setElement( 0, 0, -1 );

   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVec, 100 );
   EXPECT_EQ( eigenvalue, -1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   MatrixTypeRMO B = { { 1.0 } };
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( B, epsilon, initialVec, 10000 );
   EXPECT_EQ( eigenvalue, 1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   B.setElement( 0, 0, -1 );

   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( B, epsilon, initialVec, 100 );
   EXPECT_EQ( eigenvalue, -1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationDense2D()
{
   const TNL::Algorithms::Segments::ElementsOrganization organization = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixType = TNL::Matrices::DenseMatrix< RealType, Device, int, organization >;
   const MatrixType A = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 1.0, 2.0 };
   const RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( A, epsilon, initialVec, 10000 );
   EXPECT_NEAR( eigenvalue, 3, 1e-7 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_NEAR( eigenvector.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   const MatrixType B = { { 0, 1 }, { 1, 0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecB = { 1.0, 2.0 };
   auto [ eigenvalueB, eigenvectorB, iterationsB ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( B, epsilon, initialVecB, 10000 );
   EXPECT_EQ( iterationsB, 0 );

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO C = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   VectorType initialVecRMO = { 1.0, 2.0 };
   auto [ eigenvalueRMO, eigenvectorRMO, iterationsRMO ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( C, epsilon, initialVecRMO, 10000 );
   EXPECT_NEAR( eigenvalueRMO, 3, 1e-7 );
   for( int i = 0; i < eigenvectorRMO.getSize(); i++ ) {
      EXPECT_NEAR( eigenvectorRMO.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   const MatrixTypeRMO D = { { 0, 1 }, { 1, 0 } };
   VectorType initialVecRMOD = { 1.0, 2.0 };
   auto [ eigenvalueRMOD, eigenvectorRMOD, iterationsRMOD ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeRMO >( D, epsilon, initialVecRMOD, 10000 );
   EXPECT_EQ( iterationsRMOD, 0 );
}

template< typename RealType, typename Device >
void
checkPowerIterationSparse0D()
{
   using MatrixTypeCMO = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   const MatrixTypeCMO A = {};
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = {};
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const MatrixTypeCMO B;

   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( B, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationExceptionSizeSquareSparse()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device >;
   MatrixType A( 1, 2 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( A, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Power iteration is possible only for square matrices", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationExceptionSizeVectorSparse()
{
   using MatrixTypeCMO = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixTypeCMO A( 1, 1 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 1 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 1 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must have the same size as the matrix", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationExceptionZeroVectorSparse()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixType A( 2, 2 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 2, 2 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 2 );
   A.setElement( 0, 1, 1 );
   A.setElement( 1, 0, 1 );
   A.setElement( 1, 1, 2 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 0, 0 };
   RealType epsilon = 1e-8;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( A, epsilon, initialVec, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must be nonzero", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationSparse1D()
{
   using MatrixTypeCMO = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixTypeCMO A( 1, 1 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 1 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 1 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 2.0 };
   RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVec, 100 );
   EXPECT_EQ( eigenvalue, 1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   A.setElement( 0, 0, -1 );
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixTypeCMO >( A, epsilon, initialVec, 100 );
   EXPECT_EQ( eigenvalue, -1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }
}

template< typename RealType, typename Device >
void
checkPowerIterationSparse2D()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixType A( 2, 2 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 2, 2 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 2 );
   A.setElement( 0, 1, 1 );
   A.setElement( 1, 0, 1 );
   A.setElement( 1, 1, 2 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 1.0, 2.0 };
   RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( A, epsilon, initialVec, 10000 );
   EXPECT_NEAR( eigenvalue, 3, 1e-7 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_NEAR( eigenvector.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   MatrixType B( 2, 2 );
   TNL::Containers::Vector< int, Device > rowCapacitiesB{ 1, 1 };
   B.setRowCapacities( rowCapacitiesB );
   B.setElement( 0, 1, 1 );
   B.setElement( 1, 0, 1 );
   VectorType initialVecB = { 1.0, 2.0 };
   auto [ eigenvalueB, eigenvectorB, iterationsB ] =
      TNL::Solvers::Eigen::experimental::powerIteration< MatrixType >( B, epsilon, initialVecB, 10000 );
   EXPECT_EQ( iterationsB, 0 );
}

TEST( PowerMethodTest, PowerIteration )
{
#if ! defined( __CUDACC__ )
   checkPowerIterationExceptionSizeSquare< double, TNL::Devices::Host >();
   checkPowerIterationExceptionSizeSquare< float, TNL::Devices::Host >();
   checkPowerIterationExceptionSizeVector< double, TNL::Devices::Host >();
   checkPowerIterationExceptionSizeVector< float, TNL::Devices::Host >();
   checkPowerIterationExceptionZeroVector< double, TNL::Devices::Host >();
   checkPowerIterationExceptionZeroVector< float, TNL::Devices::Host >();
   checkPowerIterationDense0D< double, TNL::Devices::Host >();
   checkPowerIterationDense0D< float, TNL::Devices::Host >();
   checkPowerIterationDense1D< double, TNL::Devices::Host >();
   checkPowerIterationDense1D< float, TNL::Devices::Host >();
   checkPowerIterationDense2D< double, TNL::Devices::Host >();
   checkPowerIterationDense2D< float, TNL::Devices::Host >();

   checkPowerIterationExceptionSizeSquareSparse< double, TNL::Devices::Host >();
   checkPowerIterationExceptionSizeSquareSparse< float, TNL::Devices::Host >();
   checkPowerIterationExceptionSizeVectorSparse< double, TNL::Devices::Host >();
   checkPowerIterationExceptionSizeVectorSparse< float, TNL::Devices::Host >();
   checkPowerIterationExceptionZeroVectorSparse< double, TNL::Devices::Host >();
   checkPowerIterationExceptionZeroVectorSparse< float, TNL::Devices::Host >();
   checkPowerIterationSparse0D< double, TNL::Devices::Host >();
   checkPowerIterationSparse0D< float, TNL::Devices::Host >();
   checkPowerIterationSparse1D< double, TNL::Devices::Host >();
   checkPowerIterationSparse1D< float, TNL::Devices::Host >();
   checkPowerIterationSparse2D< double, TNL::Devices::Host >();
   checkPowerIterationSparse2D< float, TNL::Devices::Host >();
#else
   checkPowerIterationExceptionSizeSquare< double, TNL::Devices::Cuda >();
   checkPowerIterationExceptionSizeSquare< float, TNL::Devices::Cuda >();
   checkPowerIterationExceptionSizeVector< double, TNL::Devices::Cuda >();
   checkPowerIterationExceptionSizeVector< float, TNL::Devices::Cuda >();
   checkPowerIterationExceptionZeroVector< double, TNL::Devices::Cuda >();
   checkPowerIterationExceptionZeroVector< float, TNL::Devices::Cuda >();
   checkPowerIterationDense0D< double, TNL::Devices::Cuda >();
   checkPowerIterationDense0D< float, TNL::Devices::Cuda >();
   checkPowerIterationDense1D< double, TNL::Devices::Cuda >();
   checkPowerIterationDense1D< float, TNL::Devices::Cuda >();
   checkPowerIterationDense2D< double, TNL::Devices::Cuda >();
   checkPowerIterationDense2D< float, TNL::Devices::Cuda >();

   checkPowerIterationExceptionSizeSquareSparse< double, TNL::Devices::Cuda >();
   checkPowerIterationExceptionSizeSquareSparse< float, TNL::Devices::Cuda >();
   checkPowerIterationExceptionSizeVectorSparse< double, TNL::Devices::Cuda >();
   checkPowerIterationExceptionSizeVectorSparse< float, TNL::Devices::Cuda >();
   checkPowerIterationExceptionZeroVectorSparse< double, TNL::Devices::Cuda >();
   checkPowerIterationExceptionZeroVectorSparse< float, TNL::Devices::Cuda >();
   checkPowerIterationSparse0D< double, TNL::Devices::Cuda >();
   checkPowerIterationSparse0D< float, TNL::Devices::Cuda >();
   checkPowerIterationSparse1D< double, TNL::Devices::Cuda >();
   checkPowerIterationSparse1D< float, TNL::Devices::Cuda >();
   checkPowerIterationSparse2D< double, TNL::Devices::Cuda >();
   checkPowerIterationSparse2D< float, TNL::Devices::Cuda >();
#endif
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationDense0D()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   const MatrixTypeCMO A = {};
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = {};
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const MatrixTypeCMO B;

   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( B, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO C = {};
   VectorType initialVecRMO = {};
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( C, epsilon, 2, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const MatrixTypeRMO D;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( D, epsilon, 2, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationExceptionSizeSquare()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   const MatrixTypeCMO A = { { 1, 2, 3 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Shifted power iteration is possible only for square matrices", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO B = { { 1, 2, 3 } };
   VectorType initialVecRMO = { 1, 1, 1 };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( B, epsilon, 2, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Shifted power iteration is possible only for square matrices", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationExceptionSizeVector()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   const MatrixTypeCMO A = { { 1 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must have the same size as the matrix", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO B = { { 1 } };
   VectorType initialVecRMO = { 1, 1 };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( B, epsilon, 2, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must have the same size as the matrix", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationExceptionZeroVector()
{
   const TNL::Algorithms::Segments::ElementsOrganization organization = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixType = TNL::Matrices::DenseMatrix< RealType, Device, int, organization >;
   const MatrixType A = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 0, 0 };
   const RealType epsilon = 1e-8;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVec, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must be nonzero", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO B = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   VectorType initialVecRMO = { 0, 0 };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( B, epsilon, 2, initialVecRMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must be nonzero", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationDense1D()
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   MatrixTypeCMO A = { { 1.0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 2.0 };
   RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVec, 100 );
   EXPECT_EQ( eigenvalue, 1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   A.setElement( 0, 0, -1 );
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVec, 100 );
   EXPECT_EQ( eigenvalue, -1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   MatrixTypeRMO B = { { 1.0 } };
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( B, epsilon, 2, initialVec, 10000 );
   EXPECT_EQ( eigenvalue, 1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   B.setElement( 0, 0, -1 );
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( B, epsilon, 2, initialVec, 10000 );
   EXPECT_EQ( eigenvalue, -1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationDense2D()
{
   const TNL::Algorithms::Segments::ElementsOrganization organization = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixType = TNL::Matrices::DenseMatrix< RealType, Device, int, organization >;
   const MatrixType A = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 1.0, 2.0 };
   const RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVec, 10000 );
   EXPECT_NEAR( eigenvalue, 3, 1e-7 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_NEAR( eigenvector.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   const MatrixType B = { { 0, 1 }, { 1, 0 } };
   VectorType initialVecB = { 0.1, 1.2 };
   auto [ eigenvalueB, eigenvectorB, iterationsB ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( B, epsilon, 2, initialVecB, 10000 );
   EXPECT_NEAR( eigenvalueB, 1, 1e-7 );
   for( int i = 0; i < eigenvectorB.getSize(); i++ ) {
      EXPECT_NEAR( eigenvectorB.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   const MatrixTypeRMO C = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   VectorType initialVecRMO = { 1.0, 2.0 };
   auto [ eigenvalueRMO, eigenvectorRMO, iterationsRMO ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( C, epsilon, 2, initialVecRMO, 10000 );
   EXPECT_NEAR( eigenvalueRMO, 3, 1e-7 );
   for( int i = 0; i < eigenvectorRMO.getSize(); i++ ) {
      EXPECT_NEAR( eigenvectorRMO.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   const MatrixTypeRMO D = { { 0, 1 }, { 1, 0 } };
   VectorType initialVecRMOD = { 1.0, 2.0 };
   auto [ eigenvalueRMOD, eigenvectorRMOD, iterationsRMOD ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeRMO >( D, epsilon, 2, initialVecRMOD, 10000 );
   EXPECT_NEAR( eigenvalueRMOD, 1, 1e-7 );
   for( int i = 0; i < eigenvectorRMOD.getSize(); i++ ) {
      EXPECT_NEAR( eigenvectorRMOD.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationSparse0D()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   const MatrixType A = {};
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = {};
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   const MatrixType B;

   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( B, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationExceptionSizeSquareSparse()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device >;
   MatrixType A( 1, 2 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Shifted power iteration is possible only for square matrices", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationExceptionSizeVectorSparse()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixType A( 1, 1 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 1 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 1 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 1, 1 };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVecCMO, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must have the same size as the matrix", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationExceptionZeroVectorSparse()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixType A( 2, 2 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 2, 2 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 2 );
   A.setElement( 0, 1, 1 );
   A.setElement( 1, 0, 1 );
   A.setElement( 1, 1, 2 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 1.0, 2.0 };
   const RealType epsilon = 1e-8;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVec, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "The initial vector must be nonzero", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationSparse1D()
{
   using MatrixTypeCMO = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixTypeCMO A( 1, 1 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 1 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 1 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVecCMO = { 2.0 };
   RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVecCMO, 100 );
   EXPECT_EQ( eigenvalue, 1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }

   A.setElement( 0, 0, -1 );
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixTypeCMO >( A, epsilon, 2, initialVecCMO, 100 );
   EXPECT_EQ( eigenvalue, -1 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_EQ( eigenvector.getElement( i ), 1 );
   }
}

template< typename RealType, typename Device >
void
checkShiftedPowerIterationSparse2D()
{
   using MatrixType = TNL::Matrices::SparseMatrix< RealType, Device, int >;
   MatrixType A( 2, 2 );
   TNL::Containers::Vector< int, Device > rowCapacities{ 2, 2 };
   A.setRowCapacities( rowCapacities );
   A.setElement( 0, 0, 2 );
   A.setElement( 0, 1, 1 );
   A.setElement( 1, 0, 1 );
   A.setElement( 1, 1, 2 );
   using VectorType = TNL::Containers::Vector< RealType, Device >;
   VectorType initialVec = { 1.0, 2.0 };
   const RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( A, epsilon, 2, initialVec, 10000 );
   EXPECT_NEAR( eigenvalue, 3, 1e-7 );
   for( int i = 0; i < eigenvector.getSize(); i++ ) {
      EXPECT_NEAR( eigenvector.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }

   MatrixType B( 2, 2 );
   TNL::Containers::Vector< int, Device > rowCapacitiesB{ 1, 1 };
   B.setRowCapacities( rowCapacitiesB );
   B.setElement( 0, 1, 1 );
   B.setElement( 1, 0, 1 );
   VectorType initialVecB = { 1.0, 2.0 };
   auto [ eigenvalueB, eigenvectorB, iterationsB ] =
      TNL::Solvers::Eigen::experimental::shiftedPowerIteration< MatrixType >( B, epsilon, 2, initialVecB, 10000 );
   EXPECT_NEAR( eigenvalueB, 1, 1e-7 );
   for( int i = 0; i < eigenvectorB.getSize(); i++ ) {
      EXPECT_NEAR( eigenvectorB.getElement( i ), TNL::sqrt( 2.0 ) / 2.0, 1e-7 );
   }
}

TEST( PowerMethodTest, ShiftedPowerIteration )
{
#if ! defined( __CUDACC__ )
   checkShiftedPowerIterationDense0D< double, TNL::Devices::Host >();
   checkShiftedPowerIterationDense0D< float, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionSizeSquare< double, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionSizeSquare< float, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionSizeVector< double, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionSizeVector< float, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionZeroVector< double, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionZeroVector< float, TNL::Devices::Host >();
   checkShiftedPowerIterationDense1D< double, TNL::Devices::Host >();
   checkShiftedPowerIterationDense1D< float, TNL::Devices::Host >();
   checkShiftedPowerIterationDense2D< double, TNL::Devices::Host >();
   checkShiftedPowerIterationDense2D< float, TNL::Devices::Host >();
   checkShiftedPowerIterationSparse0D< double, TNL::Devices::Host >();
   checkShiftedPowerIterationSparse0D< float, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionSizeVectorSparse< double, TNL::Devices::Host >();
   checkShiftedPowerIterationExceptionSizeVectorSparse< float, TNL::Devices::Host >();
   checkShiftedPowerIterationSparse1D< double, TNL::Devices::Host >();
   checkShiftedPowerIterationSparse1D< float, TNL::Devices::Host >();
   checkShiftedPowerIterationSparse2D< double, TNL::Devices::Host >();
   checkShiftedPowerIterationSparse2D< float, TNL::Devices::Host >();
#else
   checkShiftedPowerIterationDense0D< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationDense0D< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionSizeSquare< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionSizeSquare< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionSizeVector< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionSizeVector< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionZeroVector< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionZeroVector< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationDense1D< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationDense1D< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationDense2D< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationDense2D< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationSparse0D< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationSparse0D< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionSizeVectorSparse< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationExceptionSizeVectorSparse< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationSparse1D< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationSparse1D< float, TNL::Devices::Cuda >();
   checkShiftedPowerIterationSparse2D< double, TNL::Devices::Cuda >();
   checkShiftedPowerIterationSparse2D< float, TNL::Devices::Cuda >();
#endif
}

#include "../../main.h"

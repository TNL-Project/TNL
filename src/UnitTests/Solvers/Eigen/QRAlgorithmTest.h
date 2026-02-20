#include <tuple>
#if defined( __CUDACC__ )
   #include <TNL/Devices/Cuda.h>
#endif
#include <TNL/Devices/Host.h>
#include <TNL/Math.h>
#include <TNL/Matrices/Factorization/QR/QR.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Solvers/Eigen/experimental/QRAlgorithm.h>

#include <gtest/gtest.h>
#include <stdexcept>

template< typename RealType, typename Device >
void
checkQRAlgorithmDense0D( const TNL::Matrices::Factorization::QR::FactorizationMethod& QRmethod )
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   MatrixTypeCMO A = {};
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeCMO >( A, epsilon, QRmethod, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   MatrixTypeCMO B;

   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeCMO >( B, epsilon, QRmethod, 10 );
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
   MatrixTypeRMO C = {};
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( C, epsilon, QRmethod, 10 );
      ADD_FAILURE();
   }
   catch( const std::invalid_argument& e ) {
      EXPECT_STREQ( "Zero-sized matrices are not allowed", e.what() );
   }
   catch( ... ) {
      ADD_FAILURE();
   }

   MatrixTypeRMO D;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( D, epsilon, QRmethod, 10 );
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
checkQRAlgorithmExceptionSizeSquare( const TNL::Matrices::Factorization::QR::FactorizationMethod& QRmethod )
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   MatrixTypeCMO A = { { 1, 2, 3 } };
   RealType epsilon = 1e-6;
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeCMO >( A, epsilon, QRmethod, 10 );
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
   MatrixTypeRMO B = { { 1, 2, 3 } };
   try {
      auto [ eigenvalue, eigenvector, iterations ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( B, epsilon, QRmethod, 10 );
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
checkQRAlgorithmDense1D( const TNL::Matrices::Factorization::QR::FactorizationMethod& QRmethod )
{
   const TNL::Algorithms::Segments::ElementsOrganization organizationCMO = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixTypeCMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationCMO >;
   MatrixTypeCMO A = { { 1.0 } };
   RealType epsilon = 1e-8;
   auto [ eigenvalue, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeCMO >( A, epsilon, QRmethod, 10000 );
   EXPECT_EQ( eigenvalue.getElement( 0, 0 ), 1 );
   EXPECT_EQ( eigenvector.getElement( 0, 0 ), 1 );

   A.setElement( 0, 0, -1 );
   std::tie( eigenvalue, eigenvector, iterations ) =
      TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeCMO >( A, epsilon, QRmethod, 10000 );
   EXPECT_EQ( eigenvalue.getElement( 0, 0 ), -1 );
   EXPECT_EQ( eigenvector.getElement( 0, 0 ), 1 );
   if( QRmethod == TNL::Matrices::Factorization::QR::FactorizationMethod::Givens ) {
      const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
      using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
      MatrixTypeRMO B = { { 1.0 } };
      std::tie( eigenvalue, eigenvector, iterations ) =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( B, epsilon, QRmethod, 10000 );
      EXPECT_EQ( eigenvalue.getElement( 0, 0 ), 1 );
      EXPECT_EQ( eigenvector.getElement( 0, 0 ), 1 );

      B.setElement( 0, 0, -1 );
      std::tie( eigenvalue, eigenvector, iterations ) =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( B, epsilon, QRmethod, 10000 );
      EXPECT_EQ( eigenvalue.getElement( 0, 0 ), -1 );
      EXPECT_EQ( eigenvector.getElement( 0, 0 ), 1 );
   }
}

template< typename RealType, typename Device >
void
checkQRAlgorithmDense2D( const TNL::Matrices::Factorization::QR::FactorizationMethod& QRmethod )
{
   const TNL::Algorithms::Segments::ElementsOrganization organization = TNL::Algorithms::Segments::ColumnMajorOrder;
   using MatrixType = TNL::Matrices::DenseMatrix< RealType, Device, int, organization >;
   MatrixType A = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   const RealType epsilon = 1e-8;
   auto [ eigenvalues, eigenvector, iterations ] =
      TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixType >( A, epsilon, QRmethod, 10000 );
   EXPECT_NEAR( eigenvalues.getElement( 0, 0 ), 3, 1e-5 );
   EXPECT_NEAR( eigenvalues.getElement( 1, 1 ), 1, 1e-5 );
   EXPECT_NEAR( eigenvector.getElement( 0, 0 ), TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
   EXPECT_NEAR( eigenvector.getElement( 1, 0 ), TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
   EXPECT_NEAR( eigenvector.getElement( 0, 1 ), -TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
   EXPECT_NEAR( eigenvector.getElement( 1, 1 ), TNL::sqrt( 2.0 ) / 2.0, 1e-5 );

   MatrixType B = { { 0.0, 1.0 }, { -1.0, 0.0 } };
   auto [ eigenvaluesB, eigenvectorB, iterationsB ] =
      TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixType >( B, epsilon, QRmethod, 10000 );
   EXPECT_EQ( iterationsB, 0 );

   const TNL::Algorithms::Segments::ElementsOrganization organizationRMO = TNL::Algorithms::Segments::RowMajorOrder;
   using MatrixTypeRMO = TNL::Matrices::DenseMatrix< RealType, Device, int, organizationRMO >;
   MatrixTypeRMO C = { { 2.0, 1.0 }, { 1.0, 2.0 } };
   if( QRmethod == TNL::Matrices::Factorization::QR::FactorizationMethod::Givens ) {
      auto [ eigenvaluesRMO, eigenvectorRMO, iterationsRMO ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( C, epsilon, QRmethod, 10000 );
      EXPECT_NEAR( eigenvaluesRMO.getElement( 0, 0 ), 3, 1e-5 );
      EXPECT_NEAR( eigenvaluesRMO.getElement( 1, 1 ), 1, 1e-5 );
      EXPECT_NEAR( eigenvectorRMO.getElement( 0, 0 ), TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
      EXPECT_NEAR( eigenvector.getElement( 1, 0 ), TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
      EXPECT_NEAR( eigenvector.getElement( 0, 1 ), -TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
      EXPECT_NEAR( eigenvector.getElement( 1, 1 ), TNL::sqrt( 2.0 ) / 2.0, 1e-5 );
   }
   else {
      try {
         auto [ eigenvalue, eigenvector, iterations ] =
            TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( C, epsilon, QRmethod, 10 );
         ADD_FAILURE();
      }
      catch( const std::invalid_argument& e ) {
         EXPECT_STREQ( "Wrong QR factorization type for dense matrix with row-major order organization.", e.what() );
      }
      catch( ... ) {
         ADD_FAILURE();
      }
   }

   MatrixTypeRMO D = { { 0.0, 1.0 }, { -1.0, 0.0 } };
   if( QRmethod == TNL::Matrices::Factorization::QR::FactorizationMethod::Givens ) {
      auto [ eigenvaluesRMO, eigenvectorSRMO, iterationsRMO ] =
         TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( D, epsilon, QRmethod, 10000 );
      EXPECT_NEAR( iterationsRMO, 0, 10000 );
   }
   else {
      try {
         auto [ eigenvaluesRMO, eigenvectorsRMO, iterationsRMO ] =
            TNL::Solvers::Eigen::experimental::QRAlgorithm< MatrixTypeRMO >( D, epsilon, QRmethod, 10 );
         ADD_FAILURE();
      }
      catch( const std::invalid_argument& e ) {
         EXPECT_STREQ( "Wrong QR factorization type for dense matrix with row-major order organization.", e.what() );
      }
      catch( ... ) {
         ADD_FAILURE();
      }
   }
}

TEST( QRAlgorithmTest, QRAlgorithm )
{
#if ! defined( __CUDACC__ )
   checkQRAlgorithmDense0D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense0D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense0D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense0D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense0D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense0D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense1D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense1D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense1D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense1D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense1D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense1D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense2D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense2D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense2D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense2D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense2D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense2D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmExceptionSizeSquare< double, TNL::Devices::Host >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmExceptionSizeSquare< float, TNL::Devices::Host >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmExceptionSizeSquare< double, TNL::Devices::Host >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmExceptionSizeSquare< float, TNL::Devices::Host >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmExceptionSizeSquare< double, TNL::Devices::Host >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmExceptionSizeSquare< float, TNL::Devices::Host >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
#else
   checkQRAlgorithmDense0D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense0D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense0D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense0D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense0D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense0D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense1D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense1D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense1D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense1D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense1D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense1D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense2D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense2D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmDense2D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense2D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmDense2D< double, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmDense2D< float, TNL::Devices::Host >( TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmExceptionSizeSquare< double, TNL::Devices::Cuda >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmExceptionSizeSquare< float, TNL::Devices::Cuda >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Givens );
   checkQRAlgorithmExceptionSizeSquare< double, TNL::Devices::Cuda >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmExceptionSizeSquare< float, TNL::Devices::Cuda >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::GramSchmidt );
   checkQRAlgorithmExceptionSizeSquare< double, TNL::Devices::Cuda >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
   checkQRAlgorithmExceptionSizeSquare< float, TNL::Devices::Cuda >(
      TNL::Matrices::Factorization::QR::FactorizationMethod::Householder );
#endif
}

#include "../../main.h"

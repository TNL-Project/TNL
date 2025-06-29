/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#pragma once

#include <assert.h>
#include <string>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <math.h>
#include <fstream>

#include <TNL/Assert.h>
#include <TNL/Solvers/Linear/detail/GEMdeviceMPI.h>
#include <TNL/Solvers/Linear/detail/GEMdevice.h>
#include <TNL/Solvers/Linear/GEM.h>
#ifdef HAVE_MPI
   #include "detail/GEMdeviceMPI.h"
#else
   #include "detail/GEMdevice.h"
#endif
#include "detail/GEMkernels.h"

namespace TNL::Solvers::Linear {

template< typename Matrix >
GEM< Matrix >::GEM( MatrixGEM& A, VectorType& b )
: A( A ),
  b( b )
{}

template< typename Matrix >
bool
GEM< Matrix >::solve( VectorType& x, int verbose )
{
   //return DeviceDependentCode::SolveGEM( *this, x, pivoting, verbose );

   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Host > ) {
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM.\n" );
      return pivoting ? this->solveWithPivoting( x, verbose ) : this->solveWithoutPivoting( x, verbose );
   }
   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Cuda > ) {
#ifdef __CUDACC__
   #ifdef HAVE_MPI
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM MPI.\n" );
      this->GEMdeviceMPI( x, verbose );
   #else
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM MPI.\n" );
      this->GEMdevice( x, verbose );
   #endif
#endif
      return true;
   }
}

template< typename Matrix >
bool
GEM< Matrix >::solveWithoutPivoting( VectorType& x, int verbose )
{
   assert( b.getSize() == x.getSize() );

   const int n = this->A.getRows();

   if( verbose > 2 )
      this->print();

   for( int k = 0; k < n; k++ ) {
      /****
       * Divide the k-th row by pivot
       */
      if( verbose > 1 )
         if( k % 10 == 0 )
            std::cout << "Elimination: " << k << "/" << n << std::endl;
      const RealType& pivot = this->A.getElement( k, k );
      if( pivot == 0.0 ) {
         std::cerr << "Zero pivot has appeared in " << k << "-th step. GEM has failed." << std::endl;
         return false;
      }
      b[ k ] /= pivot;
      for( int j = k + 1; j < n; j++ )
         this->A.setElement( k, j, this->A.getElement( k, j ) / pivot );
      this->A.setElement( k, k, 1.0 );

      if( verbose > 2 ) {
         std::cout << "Dividing by the pivot ... " << std::endl;
         this->print();
      }

      /****
       * Subtract the k-th row from the rows bellow
       */
      for( int i = k + 1; i < n; i++ ) {
         for( int j = k + 1; j < n; j++ )
            this->A.setElement( i, j, this->A.getElement( i, j ) - this->A.getElement( i, k ) * this->A.getElement( k, j ) );
         b[ i ] -= this->A.getElement( i, k ) * b[ k ];
         this->A.setElement( i, k, 0.0 );
      }

      if( verbose > 2 ) {
         std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
         this->print();
      }
   }

   /****
    * Backward substitution
    */
   for( int k = n - 1; k >= 0; k-- ) {
      //if( k % 10 == 0 )
      //   std::cout << "Substitution: " << k << "/" << n << std::endl;
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A.getElement( k, j );
   }
   return true;
}

template< typename Matrix >
bool
GEM< Matrix >::solveWithPivoting( VectorType& x, int verbose )
{
   assert( b.getSize() == x.getSize() );

   const int n = this->A.getRows();

   std::cout << "A = " << A << std::endl;

   if( verbose > 2 )
      this->print();

   for( int k = 0; k < n; k++ ) {
      if( verbose > 1 )
         std::cout << "Step " << k << "/" << n << "....\n";
      /****
       * Find the pivot - the largest in k-th row
       */
      int pivotPosition( k );
      for( int i = k + 1; i < n; i++ )
         if( TNL::abs( this->A.getElement( i, k ) ) > TNL::abs( this->A.getElement( pivotPosition, k ) ) )
            pivotPosition = i;

      /****
       * Swap the rows ...
       */
      if( pivotPosition != k ) {
         for( int j = k; j < n; j++ ) {
            RealType pom = this->A.getElement( k, j );
            this->A.setElement( k, j, this->A.getElement( pivotPosition, j ) );
            this->A.setElement( pivotPosition, j, pom );
         }
         RealType pom = b[ k ];
         b[ k ] = b[ pivotPosition ];
         b[ pivotPosition ] = pom;
      }

      if( verbose > 1 ) {
         std::cout << std::endl;
         std::cout << "Choosing element at " << pivotPosition << "-th row as pivot with value " << this->A.getElement( k, k )
                   << "..." << std::endl;
         std::cout << "Swapping " << k << "-th and " << pivotPosition << "-th rows ... " << std::endl;
      }

      /****
       * Divide the k-th row by pivot
       */
      const RealType& pivot = this->A.getElement( k, k );
      if( pivot == 0.0 ) {
         std::cerr << "Zero pivot has appeared in " << k << "-th step. GEM has failed." << std::endl;
         return false;
      }
      b[ k ] /= pivot;
      for( int j = k + 1; j < n; j++ )
         this->A.setElement( k, j, this->A.getElement( k, j ) / pivot );
      this->A.setElement( k, k, 1.0 );

      if( verbose > 2 ) {
         std::cout << "Dividing by the pivot ... " << std::endl;
         this->print();
      }

      /****
       * Subtract the k-th row from the rows bellow
       */
      for( int i = k + 1; i < n; i++ ) {
         for( int j = k + 1; j < n; j++ )
            this->A.setElement( i, j, this->A.getElement( i, j ) - this->A.getElement( i, k ) * this->A.getElement( k, j ) );
         b[ i ] -= this->A.getElement( i, k ) * b[ k ];
         this->A.setElement( i, k, 0.0 );
      }

      if( verbose > 2 ) {
         std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
         this->print();
      }
   }

   /****
    * Backward substitution
    */
   for( int k = n - 1; k >= 0; k-- ) {
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A.getElement( k, j );
   }
   return true;
}

template< typename Matrix >
bool
GEM< Matrix >::computeLUDecomposition( int verbose )
{
   const IndexType n = this->A.getRows();

   if( verbose > 1 )
      this->print();

   for( int k = 0; k < n; k++ ) {
      /****
       * Divide the k-th row by pivot
       */
      const RealType pivot = this->A.getElement( k, k );
      b[ k ] /= pivot;
      for( int j = k + 1; j < n; j++ )
         this->A.setElement( k, j, this->A.getElement( k, j ) / pivot );
      //A( k, k ) = 1.0;

      if( verbose > 1 ) {
         std::cout << "Dividing by the pivot ... " << std::endl;
         this->print();
      }

      /****
       * Subtract the k-th row from the rows bellow
       */
      for( int i = k + 1; i < n; i++ ) {
         for( int j = k + 1; j < n; j++ )
            this->A.setElement( i, j, this->A.getElement( i, j ) - this->A.getElement( i, k ) * this->A.getElement( k, j ) );
         b[ i ] -= this->A.getElement( i, k ) * b[ k ];
         //A( i, k ) = 0.0;
      }

      if( verbose > 1 ) {
         std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
         this->print();
      }
   }
   return true;
}

template< typename Matrix >
bool
GEM< Matrix >::GEMdevice( VectorType& x, int verbose )
{
   //Matrix* devMat = TNL::Cuda::passToDevice( this->A );
   auto matrix_view = this->A.getView();
   //TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector( this->b );

   // FOR PIVOTING SET VARIABLES ON DEVICE
   size_t size = sizeof( int );
   int* pivot;
   cudaMalloc( &pivot, size );
   int* pom = (int*) malloc( size );
   *pom = -1;

   if( verbose > 2 ) {
      // clang-format off
      detail::showMatrix<<< 1, 1 >>>( this->A.getView() );
      // clang-format on
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
      printf( "\n" );
   }

   for( int colPointer = 0; colPointer < this->A.getColumns(); colPointer++ ) {
      if( verbose > 1 )
         printf( "Elimination: %d/%d\n", colPointer, this->A.getColumns() );

      if( verbose > 2 ) {
         // clang-format off
         detail::showMatrix<<< 1, 1 >>>( this->A.getView() );
         // clang-format on
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }

      if( pivoting ) {
         // PIVOTING
         int reduceBlockSize =
            ( this->A.getColumns() - colPointer ) > 256 ? 256 : TNL::roundToMultiple( this->A.getColumns() - colPointer, 32 );
         int reduceGridSize = TNL::roundUpDivision( this->A.getColumns() - colPointer, reduceBlockSize );
         int reduceGridSizeRound = TNL::roundToMultiple( reduceGridSize, 32 );
         //printf("%d, %d, %d\n", reduceBlockSize, reduceGridSize, reduceGridSizeRound );

         VectorType outMax( reduceGridSize );
         TNL::Containers::Vector< IndexType, TNL::Devices::Cuda, IndexType > outPos( reduceGridSize );
         //outMax.setValue(0); outPos.setValue(0);

         // clang-format off
         detail::findPivot<<< reduceGridSize, reduceBlockSize >>>( matrix_view, colPointer, outMax.getView(), outPos.getView() );
         // clang-format on
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;

         // clang-format off
         detail::findRowPivot<<< 1, reduceGridSizeRound >>>( outMax.getView(), outPos.getView(), pivot );
         // clang-format on
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
         *pom = 0;
         cudaMemcpy( pom, pivot, size, cudaMemcpyDeviceToHost );
      }

      int blockSize = this->A.getRows() * ( this->A.getColumns() - colPointer ) > 256
                       ? 256
                       : this->A.getRows() * ( this->A.getColumns() - colPointer );
      int gridSize = TNL::roundUpDivision( this->A.getRows() * ( this->A.getColumns() - colPointer ), blockSize );
      //printf( "%d: %d, %d\n", colPointer, blockSize, numOfBlocks );

      if( pivoting )  // && *pom != -1 && *pom != colPointer )
      {
         if( verbose > 1 ) {
            std::cout << std::endl;
            std::cout << "Choosing element at " << *pom << "-th row as pivot with value..." << std::endl;
            std::cout << "Swapping " << colPointer << "-th and " << *pom << "-th rows ... " << std::endl;
         }
         int blockSizeSwap = this->A.getColumns() - colPointer > 256 ? 256 : this->A.getColumns() - colPointer;
         int gridSizeSwap = TNL::roundUpDivision( this->A.getColumns() - colPointer, blockSize );

         // clang-format off
         detail::swapRows<<< gridSizeSwap, blockSizeSwap >>>( matrix_view, this->b.getView(), colPointer, pivot );
         // clang-format on
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }

      // clang-format off
      detail::GEMmainKernel<<< gridSize, blockSize >>>( matrix_view, this->b.getView(), colPointer );
      // clang-format on
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
   }

   cudaFree( pivot );
   free( pom );

   detail::calculResultVector( this->A, this->b, x );

   return true;
}

template< typename Matrix >
void
GEM< Matrix >::print( std::ostream& str ) const
{
   const IndexType n = A.getRows();
   const int precision( 18 );
   const std::string zero( "." );
   for( int row = 0; row < n; row++ ) {
      str << "| ";
      for( int column = 0; column < n; column++ ) {
         const RealType value = this->A.getElement( row, column );
         if( value == 0.0 )
            str << std::setw( precision + 6 ) << zero;
         else
            str << std::setprecision( precision ) << std::setw( precision + 6 ) << value;
      }
      str << " | " << std::setprecision( precision ) << std::setw( precision + 6 ) << b.getElement( row ) << " |" << std::endl;
   }
}

/*template<>
class GEMDeviceDependentCode< TNL::Devices::Host >
{
public:
   typedef TNL::Devices::Host DeviceType;

   template< typename Real, typename Index >
   static bool
   SolveGEM( GEM< Real, DeviceType, Index >& gem,
             TNL::Containers::Vector< Real, DeviceType, Index >& x,
             const TNL::String& pivoting,
             int verbose )
   {
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM.\n" );
      return pivoting == "yes" ? gem.solveWithPivoting( x, verbose ) : gem.solveWithoutPivoting( x, verbose );
   }
};

template<>
class GEMDeviceDependentCode< TNL::Devices::Cuda >
{
public:
   typedef TNL::Devices::Cuda DeviceType;

   template< typename Real, typename Index >
   static bool
   SolveGEM( GEM< Real, DeviceType, Index >& gem,
             TNL::Containers::Vector< Real, DeviceType, Index >& x,
             const TNL::String& pivoting,
             int verbose )
   {
#ifdef HAVE_CUDA
   #ifdef HAVE_MPI
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM MPI.\n" );
      gem.GEMdeviceMPI( x, pivoting, verbose );
   #else
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM MPI.\n" );
      gem.GEMdevice( x, pivoting, verbose );
   #endif
#endif
      return true;
   }
};*/

}  // namespace TNL::Solvers::Linear

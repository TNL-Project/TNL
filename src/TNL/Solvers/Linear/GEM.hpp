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
#include <TNL/Containers/StaticVector.h>
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
   using CoordinateType = typename Containers::StaticVector< 2, IndexType >;
   TNL_ASSERT_EQ( b.getSize(), x.getSize(), "The sizes of of vectors x and b do not match." );

   const int n = this->A.getRows();
   auto matrix_view = this->A.getView();
   auto b_view = b.getView();

   std::cout << "Matrix A: \n" << A << std::endl;
   std::cout << "Vector b: \n" << b << std::endl;
   for( int k = 0; k < n; k++ ) {
      // Find the pivot - the largest in k-th row
      auto [ pivot_value, pivot_position ] = Algorithms::reduceWithArgument< DeviceType >(
         k,
         n,
         [ = ] __cuda_callable__( const IndexType rowIdx ) -> RealType
         {
            return abs( matrix_view( rowIdx, k ) );
         },
         TNL::MaxWithArg{} );
      if( pivot_value == 0.0 )
         throw std::runtime_error( "Zero pivot has appeared in step " + convertToString( k ) + ". GEM has failed." );

      // Swap the rows ...
      if( pivot_position != k ) {
         Algorithms::parallelFor< DeviceType >( k,
                                                n,
                                                [ = ] __cuda_callable__( const IndexType i ) mutable
                                                {
                                                   swap( matrix_view( k, i ), matrix_view( pivot_position, i ) );
                                                   if( i == k ) {
                                                      swap( b_view[ k ], b_view[ pivot_position ] );
                                                   }
                                                } );
      }

      /*b[ k ] /= pivot_value;
      for( int j = k + 1; j < n; j++ )
         this->A( k, j ) /= pivot_value;
      this->A( k, k ) = 1.0;*/

      Algorithms::parallelFor< DeviceType >( k,
                                             n,
                                             [ = ] __cuda_callable__( const IndexType i ) mutable
                                             {
                                                if( i == k ) {
                                                   matrix_view( k, i ) = 1.0;
                                                   b[ k ] /= pivot_value;
                                                }
                                                else
                                                   matrix_view( k, i ) /= pivot_value;
                                             } );

      //if( verbose > 2 ) {
      //   std::cout << "Dividing by the pivot ... " << std::endl;
      //   this->print();
      //}

      Algorithms::parallelFor< DeviceType >( CoordinateType{ 0, k },
                                             CoordinateType{ n, n },
                                             [ = ] __cuda_callable__( const CoordinateType c ) mutable
                                             {
                                                const auto& i = c[ 0 ];
                                                const auto& j = c[ 1 ];
                                                if( i != k ) {
                                                   // Subtract the k-th row from the current row
                                                   if( j > k )
                                                      matrix_view( i, j ) -= matrix_view( i, k ) * matrix_view( k, j );
                                                   else
                                                      b[ i ] -= matrix_view( i, k ) * b[ k ];
                                                }
                                             } );
      Algorithms::parallelFor< DeviceType >( 0,
                                             n,
                                             [ = ] __cuda_callable__( const IndexType i ) mutable
                                             {
                                                if( i != k ) {
                                                   //b[ i ] -= matrix_view( i, k ) * b[ k ];
                                                   matrix_view( i, k ) = 0.0;
                                                }
                                             } );

      std::cout << "A = \n" << this->A << std::endl;
      std::cout << "b = " << b << std::endl;
      // Subtract the k-th row from the rows bellow
      /*for( int i = k + 1; i < n; i++ ) {
         for( int j = k + 1; j < n; j++ )
            this->A( i, j ) = this->A( i, j ) - this->A( i, k ) * this->A( k, j );
         b[ i ] -= this->A( i, k ) * b[ k ];
         this->A( i, k ) = 0.0;
      }*/

      //if( verbose > 2 ) {
      //   std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
      //   this->print();
      //}
   }

   // Backward substitution
   /*for( int k = n - 1; k >= 0; k-- ) {
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A( k, j );
   }*/
   x = b;
   return true;

   /*if constexpr( std::is_same_v< DeviceType, TNL::Devices::Host > ) {
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
         printf( "Starting the computation SolveGEM on GPU.\n" );
      this->GEMdevice( x, verbose );
   #endif
   #endif
      return true;
   }*/
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
      // Divide the k-th row by pivot
      //if( verbose > 1 )
      //   if( k % 10 == 0 )
      //      std::cout << "Elimination: " << k << "/" << n << std::endl;
      const RealType& pivot = this->A( k, k );
      if( pivot == 0.0 ) {
         std::cerr << "Zero pivot has appeared in step " << k << ". GEM has failed." << std::endl;
         return false;
      }
      b[ k ] /= pivot;
      for( int j = k + 1; j < n; j++ )
         this->A( k, j ) /= pivot;
      this->A( k, k ) = 1.0;

      //if( verbose > 2 ) {
      //   std::cout << "Dividing by the pivot ... " << std::endl;
      //   this->print();
      //}

      // Subtract the k-th row from the rows bellow
      for( int i = k + 1; i < n; i++ ) {
         for( int j = k + 1; j < n; j++ )
            this->A( i, j ) = this->A( i, j ) - this->A( i, k ) * this->A( k, j );
         b[ i ] -= this->A( i, k ) * b[ k ];
         this->A( i, k ) = 0.0;
      }

      //if( verbose > 2 ) {
      //   std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
      //   this->print();
      //}
   }

   // Backward substitution
   for( int k = n - 1; k >= 0; k-- ) {
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A( k, j );
   }
   return true;
}

template< typename Matrix >
bool
GEM< Matrix >::solveWithPivoting( VectorType& x, int verbose )
{
   assert( b.getSize() == x.getSize() );

   const int n = this->A.getRows();
   auto matrix_view = this->A.getView();
   auto b_view = b.getView();

   for( int k = 0; k < n; k++ ) {
      // Find the pivot - the largest in k-th row
      //IndexType pivotPosition( k );
      //for( IndexType i = k + 1; i < n; i++ )
      //   if( TNL::abs( this->A( i, k ) ) > TNL::abs( this->A( pivotPosition, k ) ) )
      //      pivotPosition = i;

      auto [ pivot_value, pivot_position ] = Algorithms::reduceWithArgument< DeviceType >(
         k,
         n,
         [ = ] __cuda_callable__( const IndexType rowIdx ) -> RealType
         {
            return abs( matrix_view( rowIdx, k ) );
         },
         TNL::MaxWithArg{} );
      if( pivot_value == 0.0 )
         throw std::runtime_error( "Zero pivot has appeared in step " + convertToString( k ) + ". GEM has failed." );

      //pivot_position += k;

      //std::cout << "Pivot position: " << pivot_position << std::endl;
      // Swap the rows ...
      if( pivot_position != k ) {
         Algorithms::parallelFor< DeviceType >( k + 1,
                                                n,
                                                [ = ] __cuda_callable__( const IndexType i ) mutable
                                                {
                                                   //RealType pom = this->A( i, k );
                                                   swap( matrix_view( i, k ), matrix_view( i, pivot_position ) );
                                                   //this->A( i, pivot_position ) = pom;
                                                   if( i == k + 1 ) {
                                                      swap( b_view[ i ], b_view[ k ] );
                                                   }
                                                } );
      }

      /*b[ k ] /= pivot_value;
      for( int j = k + 1; j < n; j++ )
         this->A( k, j ) /= pivot_value;
      this->A( k, k ) = 1.0;*/

      Algorithms::parallelFor< DeviceType >( k,
                                             n,
                                             [ = ] __cuda_callable__( const IndexType i ) mutable
                                             {
                                                if( i == k ) {
                                                   matrix_view( i, k ) = 1.0;
                                                   b[ k ] /= pivot_value;
                                                }
                                                else
                                                   matrix_view( i, k ) /= pivot_value;
                                             } );

      //if( verbose > 2 ) {
      //   std::cout << "Dividing by the pivot ... " << std::endl;
      //   this->print();
      //}

      // Subtract the k-th row from the rows bellow
      for( int i = k + 1; i < n; i++ ) {
         for( int j = k + 1; j < n; j++ )
            this->A( i, j ) = this->A( i, j ) - this->A( i, k ) * this->A( k, j );
         b[ i ] -= this->A( i, k ) * b[ k ];
         this->A( i, k ) = 0.0;
      }

      //if( verbose > 2 ) {
      //   std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
      //   this->print();
      //}
   }

   // Backward substitution
   for( int k = n - 1; k >= 0; k-- ) {
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A( k, j );
   }
   return true;
}

template< typename Matrix >
bool
GEM< Matrix >::computeLUDecomposition( int verbose )
{
   const IndexType n = this->A.getRows();

   //if( verbose > 1 )
   //   this->print();

   for( int k = 0; k < n; k++ ) {
      // Divide the k-th row by pivot
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
#ifdef __CUDACC__
   auto matrix_view = this->A.getView();
   const IndexType n = this->A.getRows();

   // FOR PIVOTING SET VARIABLES ON DEVICE
   //size_t size = sizeof( int );
   IndexType pivot;
   //cudaMalloc( &pivot, sizeof( IndexType ) );
   //IndexType pom = -1;

   //if( verbose > 2 ) {
   //   // clang-format off
   //   detail::showMatrix<<< 1, 1 >>>( this->A.getView() );
   //   // clang-format on
   //   cudaDeviceSynchronize();
   //   TNL_CHECK_CUDA_DEVICE;
   //   printf( "\n" );
   //}

   for( IndexType k = 0; k < n; k++ ) {
      //if( verbose > 1 )
      //   printf( "Elimination: %d/%d\n", colPointer, this->A.getColumns() );

      //if( verbose > 2 ) {
      //   // clang-format off
      //   detail::showMatrix<<< 1, 1 >>>( this->A.getView() );
      //   // clang-format on
      //   cudaDeviceSynchronize();
      //   TNL_CHECK_CUDA_DEVICE;
      //}

      if( pivoting ) {
         /*VectorType aux( this->A.getRows() - k );
         aux.forAllElements(
            [ = ] __cuda_callable__( const IndexType i, RealType& value )
            {
               value = abs( matrix_view( i + k, k ) );
            } );*/
         auto [ pivot_value, pivot_position ] = Algorithms::reduceWithArgument< DeviceType >(
            k,
            n,
            [ = ] __cuda_callable__( const IndexType rowIdx ) -> RealType
            {
               return abs( matrix_view( rowIdx, k ) );
            },
            TNL::MaxWithArg{} );
         if( pivot_value == 0.0 )
            throw std::runtime_error( "Zero pivot has appeared in step " + convertToString( k ) + ". GEM has failed." );

         pivot = pivot_position + k;
         //cudaMemcpy( &pivot, &pom, sizeof( IndexType ), cudaMemcpyHostToDevice );

         /*IndexType reducedBlockSize =
            ( this->A.getColumns() - colPointer ) > 256 ? 256 : TNL::roundToMultiple( this->A.getColumns() - colPointer, 32 );
         IndexType reducedGridSize = TNL::roundUpDivision( this->A.getColumns() - colPointer, reducedBlockSize );
         IndexType reducedGridSizeRound = TNL::roundToMultiple( reducedGridSize, 32 );

         VectorType outMax( reducedGridSize );
         TNL::Containers::Vector< IndexType, TNL::Devices::Cuda, IndexType > outPos( reducedGridSize );

         Devices::Cuda::LaunchConfiguration launch_config;
         launch_config.blockSize.x = reducedBlockSize;
         launch_config.gridSize.x = reducedGridSize;
         auto findPivot_kernel =
            detail::findPivot< decltype( matrix_view ), decltype( outMax.getView() ), decltype( outPos.getView() ) >;
         Backend::launchKernel( findPivot_kernel, launch_config, matrix_view, colPointer, outMax.getView(), outPos.getView() );
         cudaDeviceSynchronize();


         launch_config.blockSize.x = reducedBlockSize;
         launch_config.gridSize.x = reducedGridSize;
         // clang-format off
         detail::findRowPivot<<< 1, reducedGridSizeRound >>>( outMax.getView(), outPos.getView(), pivot );
         // clang-format on
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
         *pom = 0;
         cudaMemcpy( pom, pivot, size, cudaMemcpyDeviceToHost );*/
      }

      int blockSize =
         this->A.getRows() * ( this->A.getColumns() - k ) > 256 ? 256 : this->A.getRows() * ( this->A.getColumns() - k );
      int gridSize = TNL::roundUpDivision( this->A.getRows() * ( this->A.getColumns() - k ), blockSize );

      if( pivoting ) {
         //if( verbose > 1 ) {
         //   std::cout << std::endl;
         //   std::cout << "Choosing element at " << *pom << "-th row as pivot with value..." << std::endl;
         //   std::cout << "Swapping " << colPointer << "-th and " << *pom << "-th rows ... " << std::endl;
         //}
         int blockSizeSwap = this->A.getColumns() - k > 256 ? 256 : this->A.getColumns() - k;
         int gridSizeSwap = TNL::roundUpDivision( this->A.getColumns() - k, blockSize );

         // clang-format off
         detail::swapRows<<< gridSizeSwap, blockSizeSwap >>>( matrix_view, this->b.getView(), k, pivot );
         // clang-format on
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }

      // clang-format off
      detail::GEMmainKernel<<< gridSize, blockSize >>>( matrix_view, this->b.getView(), k );
      // clang-format on
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
   }

   //cudaFree( pivot );
   //free( pom );

   detail::calculResultVector( this->A, this->b, x );
#endif
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

}  // namespace TNL::Solvers::Linear

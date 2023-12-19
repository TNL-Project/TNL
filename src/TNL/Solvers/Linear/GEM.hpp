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
//#ifdef HAVE_MPI
//   #include "GEMdeviceMPI.h"
//#else
//   #include "GEMdevice.h"
//#endif
#include <TNL/Solvers/Linear/GEM.h>

namespace TNL::Solvers::Linear {

template< typename Matrix >
GEM< Matrix >::GEM( MatrixGEM& A, VectorType& b ) : A( A ), b( b )
{}

template< typename Matrix >
bool
GEM< Matrix >::solve( VectorType& x, const TNL::String& pivoting, int verbose )
{
   //return DeviceDependentCode::SolveGEM( *this, x, pivoting, verbose );

   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Host > ) {
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM.\n" );
      return pivoting == "yes" ? this->solveWithPivoting( x, verbose ) : this->solveWithoutPivoting( x, verbose );
   }
   if constexpr( std::is_same_v< DeviceType, TNL::Devices::Cuda > ) {
#ifdef __CUDACC__
   #ifdef HAVE_MPI
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM MPI.\n" );
      this->GEMdeviceMPI( x, pivoting, verbose );
   #else
      if( verbose > 1 )
         printf( "Starting the computation SolveGEM MPI.\n" );
      this->GEMdevice( x, pivoting, verbose );
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
   const IndexType n = this->A.getNumRows();

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
      str << " | " << std::setprecision( precision ) << std::setw( precision + 6 ) << b[ row ] << " |" << std::endl;
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

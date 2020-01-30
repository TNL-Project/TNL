/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <assert.h>
#include <string>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <math.h>

#pragma once

#include <TNL/Assert.h>
#include <fstream>
  
template< typename Real,
          typename Device,
          typename Index >
GEM< Real, Device, Index >::GEM( MatrixGEM& A, Array& b )
: A(A), b(b)
{}


template< typename Real,
          typename Device,
          typename Index >
bool GEM< Real, Device, Index >::solve( Array& x, int verbose )
{
   assert( b.getSize() == x.getSize() );
   
   const int n = this->A.getNumRows();

   if( verbose )
      this->print();

   for( int k = 0; k < n; k++ )
   {
      /****
       * Divide the k-th row by pivot
       */
      if( k % 10 == 0 )
         std::cout << "Elimination: " << k << "/" << n << std::endl;
      const Real& pivot = this->A.getElement( k, k );
      if( pivot == 0.0 )
      {
         std::cerr << "Zero pivot has appeared in " << k << "-th step. GEM has failed." << std::endl;
         return false;
      }
      b[ k ] /= pivot;
      for( int j = k+1; j < n; j++ )
         this->A.setElement( k, j, this->A.getElement( k, j )/pivot );
      this->A.setElement( k, k, 1.0 );
      
      if( verbose > 1 )
      {
         std::cout << "Dividing by the pivot ... " << std::endl;
         this->print();
      }
      
      /****
       * Subtract the k-th row from the rows bellow
       */
      for( int i = k+1; i < n; i++ )
      {
         for( int j = k+1; j < n; j++ )
            this->A.setElement( i, j, this->A.getElement( i, j ) - this->A.getElement( i, k ) * this->A.getElement( k, j ) );
         b[ i ] -= this->A.getElement( i, k ) * b[ k ];
         this->A.setElement( i, k, 0.0 );
      }

      if( verbose > 1 )
      {
         std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
         this->print();
      }
   }
   
   /****
    * Backward substitution
    */
   for( int k = n - 1; k >= 0; k-- )
   {
      //if( k % 10 == 0 )
      //   std::cout << "Substitution: " << k << "/" << n << std::endl;
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A.getElement( k, j );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool GEM< Real, Device, Index >::solveWithPivoting( Array& x, int verbose )
{
   assert( b.getSize() == x.getSize() );
   
   const int n = this->A.getNumRows();

   if( verbose )
      this->print();

   for( int k = 0; k < n; k++ )
   {
      std::cout << "Step " << k << "/" << n << ".... \r";
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
      if( pivotPosition != k )
      {
         for( int j = k; j < n; j++ ){
             Real pom = this->A.getElement( k, j );
             this->A.setElement( k, j, this->A.getElement( pivotPosition, j ) );
             this->A.setElement( pivotPosition, j, pom );
         }
         Real pom = b[ k ];
         b[ k ] = b[ pivotPosition];
         b[ pivotPosition ] = pom;
      }
      
      if( verbose > 1 )
      {
         std::cout << std::endl;
         std::cout << "Choosing element at " << pivotPosition << "-th row as pivot..." << std::endl;
         std::cout << "Swaping " << k << "-th and " << pivotPosition <<  "-th rows ... " << std::endl;
      }
            
      /****
       * Divide the k-th row by pivot
       */
      const Real& pivot = this->A.getElement( k, k );
      if( pivot == 0.0 )
      {
         std::cerr << "Zero pivot has appeared in " << k << "-th step. GEM has failed." << std::endl;
         return false;
      }      
      b[ k ] /= pivot;
      for( int j = k+1; j < n; j++ )
         this->A.setElement( k, j, this->A.getElement( k, j)/pivot );
      this->A.setElement( k, k, 1.0 );
      
      if( verbose > 1 )
      {
         std::cout << "Dividing by the pivot ... " << std::endl;
         this->print();
      }
      
      /****
       * Subtract the k-th row from the rows bellow
       */
      for( int i = k+1; i < n; i++ )
      {
         for( int j = k+1; j < n; j++ )
            this->A.setElement( i, j, this->A.getElement( i, j ) - this->A.getElement( i, k ) * this->A.getElement( k, j ) );
         b[ i ] -= this->A.getElement( i, k ) * b[ k ];
         this->A.setElement( i, k , 0.0 );
      }

      if( verbose > 1 )
      {
         std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
         this->print();
      }
   }
   
   /****
    * Backward substitution
    */
   for( int k = n - 1; k >= 0; k-- )
   {
      x[ k ] = b[ k ];
      for( int j = k + 1; j < n; j++ )
         x[ k ] -= x[ j ] * this->A.getElement( k, j );         
   } 
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool GEM< Real, Device, Index >::computeLUDecomposition( int verbose )
{  
   const Index n = this->A.getNumRows();

   if( verbose )
      this->print();

   for( int k = 0; k < n; k++ )
   {
      /****
       * Divide the k-th row by pivot
       */
      const Real pivot = this->A.getElement( k, k );
      b[ k ] /= pivot;
      for( int j = k+1; j < n; j++ )
         this->A.setElement( k, j, this->A.getElement( k, j )/pivot );
      //A( k, k ) = 1.0;
      
      if( verbose > 1 )
      {
         std::cout << "Dividing by the pivot ... " << std::endl;
         this->print();
      }
      
      /****
       * Subtract the k-th row from the rows bellow
       */
      for( int i = k+1; i < n; i++ )
      {
         for( int j = k+1; j < n; j++ )
            this->A.setElement( i, j, this->A.getElement( i, j ) - this->A.getElement( i, k ) * this->A.getElement( k, j ) );
         b[ i ] -= this->A.getElement( i, k ) * b[ k ];
         //A( i, k ) = 0.0;
      }

      if( verbose > 1 )
      {
         std::cout << "Subtracting the " << k << "-th row from the rows bellow ... " << std::endl;
         this->print();
      }
   }   
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void GEM< Real, Device, Index >::print( std::ostream& str ) const
{
   const Index n = A.getNumRows();
   const int precision( 18 );
   const std::string zero( "." );
   for( int row = 0; row < n; row++ )
   {
      str << "| ";
      for( int column = 0; column < n; column++ )
      {
         const Real value = this->A.getElement( row, column );
         if( value == 0.0 )
            str << std::setw( precision + 6 ) << zero;
         else str<< std::setprecision( precision ) << std::setw( precision + 6 )  << value;
      }
      str << " | " << std::setprecision( precision ) << std::setw( precision + 6 ) << b[ row ] << " |" << std::endl;
   }
}
#ifdef HAVE_CUDA
/*template < typename Real, 
           typename Index >*/
__global__ void GEMKernel( /*Matrix< Real, Devices::Cuda, Index >& matrix,
                      Containers::Array< Real, Devices::Cuda, Index >& b, 
                      Containers::Array< Real, Devices::Cuda, Index >& x, 
                      int verbose*/ )
{
  unsigned int i = threadIdx.x;// + blockDim.x * blockIdx.x;
  printf( "%d\n" , i );
};

#endif

  
/*template< typename Real,
          typename Device,
          typename Index >
bool Matrix< Real, Device, Index >::solveCuda( Array& b, Array& x, int verbose )
{
#ifdef HAVE_CUDA
  Index numRows = this->A.getNumRows();
  Index numColumns = this->A.getNumColumns();
  Index numThreads = 32;
  Index numBlocks = roundUpDivision( numRows, numThreads );
  // TODO: num of blocks needed higher than num of blocks able to calculate
  Devices::Cuda::synchronizeDevice();
  GEMKernel<<< 1, 1 >>>( this, b, x, verbose );
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
#endif
}*/

#pragma once

#ifdef __CUDACC__
   #include <cuda_runtime.h>

namespace TNL::Solvers::Linear::detail {

template< typename Real >
__inline__ __device__
void
warpReduceArgMax( Real& val, int& index )
{
   __syncthreads();
   for( int offset = 32 / 2; offset > 0; offset /= 2 ) {
      Real val1 = __shfl_down_sync( 0xffffffff, val, offset, 32 );
      int index1 = __shfl_down_sync( 0xffffffff, index, offset, 32 );
      __syncthreads();
      if( TNL::abs( val1 ) - TNL::abs( val ) > 0 ) {
         val = val1;
         index = index1;
      }
      __syncthreads();
   }
}

template< typename Real >
__inline__ __device__
void
blockReduceArgMax( Real& val, int& index )
{
   static __shared__ Real sharedVal[ 32 ];    // Shared mem for 32 partial reduction
   static __shared__ Real sharedIndex[ 32 ];  // Shared mem for 32 partial reduction
   int lane = threadIdx.x % 32;
   int wid = threadIdx.x / 32;

   warpReduceArgMax( val, index );  // Each warp performs partial reduction

   if( lane == 0 ) {
      sharedVal[ wid ] = val;      // Write reduced value to shared memory
      sharedIndex[ wid ] = index;  // Write reduced value to shared memory
   }

   __syncthreads();  // Wait for all partial reductions

   //read from shared memory only if that warp existed
   val = ( threadIdx.x < blockDim.x / 32 ) ? sharedVal[ lane ] : 0;
   index = ( threadIdx.x < blockDim.x / 32 ) ? sharedIndex[ lane ] : 0;

   if( wid == 0 ) {
      warpReduceArgMax( val, index );  //Final reduce within first warp
   }
}

template< typename MatrixView, typename VectorView, typename IndexVectorView >
__global__
void
findPivot( MatrixView A, int colPointerMain, VectorView outMaximum, IndexVectorView outPosition )
{
   using Real = typename MatrixView::RealType;
   using Index = typename MatrixView::IndexType;

   int rowPointer = threadIdx.x + blockDim.x * blockIdx.x + colPointerMain;
   Real firstElementInRow = rowPointer >= A.getRows() ? 0 : TNL::abs( A.getElement( rowPointer, colPointerMain ) );
   int index = rowPointer;
   blockReduceArgMax( firstElementInRow, index );
   if( threadIdx.x == 0 ) {
      outMaximum[ blockIdx.x ] = firstElementInRow;
      outPosition[ blockIdx.x ] = index;
   }
}

template< typename VectorView, typename IndexVectorView >
__global__
void
findRowPivot( VectorView outMaximum, IndexVectorView outPosition, int* positionPivot )
{
   using Real = typename VectorView::RealType;
   using Index = typename VectorView::IndexType;

   int rowPointer = threadIdx.x;
   Real firstElementInRow = rowPointer >= outMaximum.getSize() ? 0 : outMaximum[ rowPointer ];
   int index = rowPointer >= outPosition.getSize() ? 0 : outPosition[ rowPointer ];
   blockReduceArgMax( firstElementInRow, index );
   if( threadIdx.x == 0 ) {
      *positionPivot = index;
   }
}

template< typename MatrixView, typename VectorView >
__global__
void
swapRows( MatrixView A, VectorView b, int colPointerMain, int* positionPivot )
{
   using Real = typename MatrixView::RealType;
   using Index = typename MatrixView::IndexType;

   if( *positionPivot > colPointerMain ) {
      int rowPointer1 = colPointerMain;
      int rowPointer2 = *positionPivot;
      int colPointer = threadIdx.x + blockDim.x * blockIdx.x + colPointerMain;
      if( colPointer < A.getColumns() && rowPointer1 < A.getRows() ) {
         Real pom = A.getElement( rowPointer1, colPointer );
         A.setElement( rowPointer1, colPointer, A.getElement( rowPointer2, colPointer ) );
         A.setElement( rowPointer2, colPointer, pom );
         if( colPointer == colPointerMain ) {
            pom = b[ rowPointer1 ];
            b[ rowPointer1 ] = b[ rowPointer2 ];
            b[ rowPointer2 ] = pom;
         }
      }
   }
}

template< typename MatrixView, typename VectorView >
__global__
void
GEMmainKernel( MatrixView A, VectorView b, int colPointerMain )
{
   using Real = typename MatrixView::RealType;
   using Index = typename MatrixView::IndexType;

   int thread = threadIdx.x + blockIdx.x * blockDim.x;
   int rowPointer = thread / ( A.getRows() - colPointerMain );
   int colPointer = thread % ( A.getRows() - colPointerMain ) + colPointerMain;
   //printf("%d, %d\n",rowPointer, colPointer );
   if( colPointer > colPointerMain && colPointer < A.getColumns() && rowPointer != colPointerMain && rowPointer < A.getRows() )
   {
      if( A.getElement( colPointerMain, colPointerMain ) != 0 ) {
         const Real pivot = A.getElement( colPointerMain, colPointerMain );
         const Real firstElementInRow = A.getElement( rowPointer, colPointerMain );
         if( firstElementInRow != 0 ) {
            A.setElement( rowPointer,
                          colPointer,
                          A.getElement( rowPointer, colPointer )
                             - firstElementInRow * A.getElement( colPointerMain, colPointer ) / pivot );
         }
      }
      else if( colPointer == colPointerMain && rowPointer == colPointerMain )
         printf( "Error, pivot is zero!\n" );
   }
   if( rowPointer < A.getRows() && colPointer < A.getColumns() && rowPointer != colPointerMain && colPointer == colPointerMain
       && A.getElement( colPointerMain, colPointerMain ) != 0 && A.getElement( rowPointer, colPointerMain ) != 0 )
   {
      b[ rowPointer ] =
         b[ rowPointer ]
         - A.getElement( rowPointer, colPointerMain ) * b[ colPointerMain ] / A.getElement( colPointerMain, colPointerMain );
   }
}

template< typename MatrixView, typename ConstVectorView, typename VectorView >
__global__
void
GEMDiagToResult( const MatrixView A, const ConstVectorView v, VectorView out )
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if( i < A.getRows() )
      out[ i ] = v[ i ] / A.getElement( i, i );
}

template< typename MatrixView >
__global__
void
showMatrix( MatrixView A )
{
   //A.showMatrix();
}

}  // namespace TNL::Solvers::Linear::detail

#endif

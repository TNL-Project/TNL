#ifdef HAVE_CUDA
#include <cuda_runtime.h>
//TODO: Real

/************************REDUCTION MAX******************************************/

__inline__ __device__ void warpReduceArgMax(float& val, int& index) {
  //printf( "index = %d\n", index );
  __syncthreads();
  for (int offset = 32/2; offset > 0; offset /= 2) 
  { 
    float val1 = __shfl_down_sync( 0xffffffff, val, offset, 32);
    int index1 = __shfl_down_sync( 0xffffffff, index, offset, 32);
    __syncthreads();
    //printf("%d: firstElementInRow = %.4f, %d: val1 = %.4f\n", index, val, index1, val1 );
    if( TNL::abs( val1 )  - TNL::abs( val ) > 0 )
    {
      val = val1;
      index = index1;
      //printf("%d: %.4f\n", index, firstElementInRow  );
    }
    __syncthreads();
    //printf("%d: firstElementInRow = %.4f\n", index, val );
  } 
}

__inline__ __device__ void blockReduceArgMax(float& val, int& index) 
{
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  warpReduceArgMax( val, index);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0)
  {
    warpReduceArgMax( val, index ); //Final reduce within first warp
  }

}



/*************************END REDUCTION MAX*************************************/
/*******************************************************************************/


__global__ 
void findPivot( Matrix< double, TNL::Devices::Cuda, int >* A, 
        int colPointerMain, TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > outMaximum,
        TNL::Containers::VectorView< int, TNL::Devices::Cuda, int > outPosition)
{
  int rowPointer = threadIdx.x + blockDim.x * blockIdx.x + colPointerMain;
  float firstElementInRow = rowPointer >= A->getNumRows() ? 0 : TNL::abs( A->getElement(rowPointer, colPointerMain) );
  int index = rowPointer;
  blockReduceArgMax( firstElementInRow, index );
  if( threadIdx.x == 0 )
  {
    //printf("%d: %.2f\n", index, firstElementInRow );
    outMaximum[blockIdx.x] = firstElementInRow;
    outPosition[blockIdx.x] = index;
  }
    
  
  //if( rowPointer < A->getNumRows() && rowPointer >= colPointerMain )
  {
    //int pom = __float_as_int( (float)TNL::abs( A->getElement( rowPointer, colPointerMain ) ) );
    /*if( TNL::abs( A->getElement(rowPointer, colPointerMain ) ) > 1 )
      printf("%d: %d, %.4f\n", rowPointer, pom, TNL::abs( A->getElement(rowPointer, colPointerMain ) ) );*/
    //atomicMax( Maximum, pom );
  } 
  
  
  //if( threadIdx.x == 0 && blockIdx.x == 0 )
  //    printf("%.4f %d\n", firstElementInRow, index );
}


__global__ 
void findRowPivot( TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > outMaximum,
        TNL::Containers::VectorView< int, TNL::Devices::Cuda, int > outPosition, int* positionPivot )
{
  /*if( threadIdx.x == 0 )
  {
    printf("outMax = [ ");
    for( int i = 0; i < blockDim.x; i++ )
      printf("%.2f, ", outMaximum[i] );
    printf("]\n");
    
    printf("outPos = [ ");
    for( int i = 0; i < blockDim.x; i++ )
      printf("%d, ", outPosition[i] );
    printf("]\n");
  }*/
  
  int rowPointer = threadIdx.x;
  float firstElementInRow = rowPointer >= blockDim.x ? 0 : outMaximum[ rowPointer ];
  int index = outPosition[ rowPointer ];
  blockReduceArgMax( firstElementInRow, index );
  //printf("%d: %.2f\n", index, firstElementInRow );
  if( threadIdx.x == 0 )
  {
    *positionPivot = index;
  }
  
  /*int rowPointer = threadIdx.x + blockDim.x * (blockIdx.x % numBlocksOnRow) + colPointerMain;
  if( rowPointer >= colPointerMain && rowPointer < A->getNumRows() &&
          __float_as_int( (float)TNL::abs( A->getElement( rowPointer, colPointerMain ) ) ) == *Maximum )
  {
    atomicExch( positionPivot, rowPointer);
  }*/
}
__global__ 
void swapRows( Matrix< double, TNL::Devices::Cuda, int >* A, 
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b,
        int colPointerMain, int numBlocksOnRow, int* positionPivot )
{
  int rowPointer1 = colPointerMain;
  int rowPointer2 = *positionPivot;
  int colPointer = threadIdx.x + blockDim.x * (blockIdx.x % numBlocksOnRow) + colPointerMain;
  if( colPointer < A->getNumColumns() && rowPointer1 < A->getNumRows() )
  {
    double pom = A->getElement( rowPointer1, colPointer );
    A->setElement( rowPointer1, colPointer, A->getElement( rowPointer2, colPointer ) );
    A->setElement( rowPointer2, colPointer, pom );
    if( colPointer == colPointerMain )
    {
      pom = b[rowPointer1];
      b[rowPointer1] = b[rowPointer2];
      b[rowPointer2] = pom;
    }
  }
}


__global__ 
void GEMColumnUnderDiag( Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b, 
        int colPointerMain, int numBlocksOnRow )
{
  int rowPointer = blockIdx.x / numBlocksOnRow;
  int colPointer = threadIdx.x + blockDim.x * (blockIdx.x % numBlocksOnRow) + colPointerMain;
  if( colPointer < A->getNumColumns() && rowPointer != colPointerMain )
  {
    if( A->getElement( colPointerMain, colPointerMain ) != 0 )
    { 
      const double pivot = A->getElement( colPointerMain, colPointerMain );
      const double firstElementInRow = A->getElement( rowPointer, colPointerMain );
      if( firstElementInRow != 0 )
      {
        A->setElement( rowPointer, colPointer,
                  A->getElement( rowPointer, colPointer ) - firstElementInRow * A->getElement( colPointerMain, colPointer ) / pivot );   
        
        if( colPointer == colPointerMain )
        {
          b[ rowPointer ] = b[ rowPointer ] - firstElementInRow * b[ colPointerMain ] / pivot;
          A->setElement( rowPointer, colPointerMain, 0.0 );
        }
      }
    } else if( colPointer == colPointerMain && rowPointer == colPointerMain ) printf( "Error, pivot is zero!\n");
  }
}


__global__ 
void GEMDiagToResult( Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > v,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > out )
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if( i < A->getNumRows() )
    out[i] = v[i] / A->getElement(i,i);
}


/*********************FIRST TRY WITH COLUMNS************************************/


__global__ 
void GEMForwardPass( Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b, int rowPointer )
{
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row < A->getNumRows() && row > rowPointer )
  {
    if( A->getElement( rowPointer, rowPointer ) != 0 )
    {    
      const double pivot = A->getElement( rowPointer, rowPointer );
      const double firstElementInRow = A->getElement( row, rowPointer );
      if( firstElementInRow != 0 )
      {
        b[ row ] = b[ rowPointer ] - pivot*b[ row ] / firstElementInRow;
        A->setElement( row, rowPointer, 0. );
        for( int i = rowPointer+1; i < A->getNumColumns(); i++ )
        {
          A->setElement( row, i,
                  A->getElement( rowPointer, i ) - pivot * A->getElement( row, i ) / firstElementInRow );
        }
      }
    } else printf( "Error, pivot is zero!\n");
  }
}



__global__
void GEMNormRows( Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b )
{
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row < A->getNumRows() )
  {
    const double firstElementInRow = A->getElement(row,row);
    for( int i = row; i < A->getNumRows(); i++ )
    {
      A->setElement( row, i, A->getElement( row, i ) / firstElementInRow );
    }
    b[ row ] = b[ row ] / firstElementInRow;
  }
}



__global__
void GEMBackwardPass( Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b, int rowPointer )
{
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row < rowPointer )
  {
#if DEBUG
    if(row == 0 )
    {
      for( int i = 0; i < A->getNumRows(); i++ )
        printf( "%.4f ", b[ i ] );
      printf("\n");
    }
#endif
    
    b[ row ] = b[ row ] - b[ rowPointer ] * A->getElement( row, rowPointer ) ;
    A->setElement( row, rowPointer, 0. );
    
#if DEBUG
    if(row == 0 )
    {
      for( int i = 0; i < A->getNumRows(); i++ )
        printf( "%.4f ", b[ i ] );
      printf("\n");
    }
#endif
  }
}

/**********************END FIRST TRY WITH COLUMNS ******************************/
/*******************************************************************************/



__global__ 
void showMatrix( Matrix< double, TNL::Devices::Cuda, int > A)
{
  A.showMatrix();
}



/****************************FIRST TRY WITH BLOCKS******************************/
__global__
void GEMBlocks( Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b, int mainBlockPointer )
{
  int mainRowForBlock = blockIdx.x*blockDim.x;
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row < A->getNumRows() && blockIdx.x >= mainBlockPointer )
  {
    for( int rowAdder = 0; rowAdder < blockDim.x; rowAdder++ )
    {
      int rowPointer = mainBlockPointer * blockDim.x + rowAdder;
      if( row > mainRowForBlock && mainRowForBlock < A->getNumRows() && rowPointer < A->getNumColumns() )
      {
        if( A->getElement( mainRowForBlock, rowPointer ) != 0 )
        {    
          const double pivot = A->getElement( mainRowForBlock, rowPointer );
          const double firstElementInRow = A->getElement( row, rowPointer );
          if( firstElementInRow != 0 )
          {
            b[ row ] = b[ mainRowForBlock ] - pivot*b[ row ] / firstElementInRow;
            A->setElement( row, rowPointer, 0. );
            for( int i = rowPointer+1; i < A->getNumColumns(); i++ )
            {
              A->setElement( row, i,
                      A->getElement( mainRowForBlock, i ) - pivot * A->getElement( row, i ) / firstElementInRow );
            }
          }
        } else printf( "Error, pivot is zero!\n");
      }
      __syncthreads();
      mainRowForBlock++;
    }

    
    for( int rowAdder = blockDim.x-1; rowAdder > -1; rowAdder-- )
    {
      mainRowForBlock--;
      int rowPointer = mainBlockPointer * blockDim.x + rowAdder;
      if( row < mainRowForBlock && mainRowForBlock < A->getNumRows() && rowPointer < A->getNumColumns() )
      {
        if( A->getElement( mainRowForBlock, rowPointer ) != 0 )
        {    
          const double pivot = A->getElement( mainRowForBlock, rowPointer );
          const double firstElementInRow = A->getElement( row, rowPointer );
          if( firstElementInRow != 0 )
          {
            b[ row ] = b[ row ] - firstElementInRow* b[ mainRowForBlock ]/pivot ;
            A->setElement( row, rowPointer, 0. );
            for( int i = rowPointer+1; i < A->getNumColumns(); i++ )
            {
              A->setElement( row, i,
                      A->getElement( row, i ) - firstElementInRow * A->getElement( mainRowForBlock, i ) / pivot );
            }
          }
        } else printf( "Error, pivot is zero!\n");
      }
      __syncthreads();
    }
  }
}

__global__
void GEMZeroing(  Matrix< double, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< double, TNL::Devices::Cuda, int > b, int mainBlockPointer )
{
  if( threadIdx.x == 0 && blockIdx.x == 1 )
    A->showMatrix();
  int mainRowForThisRow = mainBlockPointer*blockDim.x + threadIdx.x;
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row > mainRowForThisRow && row < A->getNumRows() && mainRowForThisRow < A->getNumColumns() && mainBlockPointer < blockIdx.x )
  {
    const double pivot = A->getElement( mainRowForThisRow, mainRowForThisRow );
    const double firstElement = A->getElement( row, mainRowForThisRow );
    A->setElement( row, mainRowForThisRow, 0. );
    b[row] = pivot * b[ row ] / firstElement - b[ mainRowForThisRow ];
    
    for( int i = mainRowForThisRow+1; i < A->getNumColumns(); i++ )
    {
      A->setElement( row, i, 
              pivot * A->getElement(row, i)/ firstElement - A->getElement(mainRowForThisRow, i ) );
    }
    
  }
}


/************************END FIRST TRY WITH BLOCKS******************************/
/*******************************************************************************/


#endif //HAVE_CUDA

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
//TODO: Real

/************************REDUCTION MAX******************************************/
template <typename Real>
__global__ void showData( Real* data, int size, int processID ){
  printf("%d: [ ", processID );
  for( int i = 0; i < size; i++ )
    printf("%.2f ", data[i] );
  printf(" ]\n");
}

template <typename Real >
__inline__ __device__ void warpReduceArgMax(Real& val, int& index) {
  __syncthreads();
  for (int offset = 32/2; offset > 0; offset /= 2) 
  { 
    Real val1 = __shfl_down_sync( 0xffffffff, val, offset, 32);
    int index1 = __shfl_down_sync( 0xffffffff, index, offset, 32);
    __syncthreads();
    if( TNL::abs( val1 )  - TNL::abs( val ) > 0 )
    {
      val = val1;
      index = index1;
    }
    __syncthreads();
  } 
}


template <typename Real >
__inline__ __device__ void blockReduceArgMax(Real& val, int& index) 
{
  static __shared__ Real sharedVal[32]; // Shared mem for 32 partial reduction
  static __shared__ Real sharedIndex[32]; // Shared mem for 32 partial reduction
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  warpReduceArgMax( val, index);     // Each warp performs partial reduction

  if (lane==0) 
  {
    sharedVal[wid]=val; // Write reduced value to shared memory
    sharedIndex[wid]=index; // Write reduced value to shared memory
  }

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / 32) ? sharedVal[lane] : 0;
  index = (threadIdx.x < blockDim.x / 32) ? sharedIndex[lane] : 0;

  if (wid==0)
  {
    warpReduceArgMax( val, index ); //Final reduce within first warp
  }

}



/*************************END REDUCTION MAX*************************************/
/*******************************************************************************/


template <typename Real >
__global__ 
void findPivot( Matrix< Real, TNL::Devices::Cuda, int >* A, 
        int fromRow, int colPointerMain, TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > outMaximum,
        TNL::Containers::VectorView< int, TNL::Devices::Cuda, int > outPosition)
{
  int rowPointer = threadIdx.x + blockDim.x * blockIdx.x + fromRow;
  Real firstElementInRow = rowPointer >= A->getNumRows() ? 0 : TNL::abs( A->getElement(rowPointer, colPointerMain) );
  int index = rowPointer;
  blockReduceArgMax( firstElementInRow, index );
  if( threadIdx.x == 0 )
  {
    outMaximum[blockIdx.x] = firstElementInRow;
    outPosition[blockIdx.x] = index;
  }
}


template <typename Real >
__global__ 
void findRowPivot( TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > outMaximum,
        TNL::Containers::VectorView< int, TNL::Devices::Cuda, int > outPosition )
{
  int rowPointer = threadIdx.x;
  Real firstElementInRow = rowPointer >= outMaximum.getSize() ? 0 : outMaximum[ rowPointer ];
  int index = rowPointer >= outPosition.getSize() ? 0 : outPosition[ rowPointer ];
  blockReduceArgMax( firstElementInRow, index );
}


template <typename Real >
__global__ 
void swapRows( Matrix< Real, TNL::Devices::Cuda, int >* A, 
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b,
        int colPointerMain, int colPointerPom, int positionPivot )
{
  if( positionPivot > colPointerPom )
  {
    int rowPointer1 = colPointerPom;
    int rowPointer2 = positionPivot;
    int colPointer = threadIdx.x + blockDim.x * blockIdx.x + colPointerMain;
    if( colPointer < A->getNumColumns() && rowPointer1 < A->getNumRows() )
    {
      Real pom = A->getElement( rowPointer1, colPointer );
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
}


template <typename Real >
__global__ 
void GEMmainKernel( Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > mainRow,
        int rowPointerMain, int colPointerMain, int processID, int numOfProcesses )
{
  int thread = threadIdx.x + blockDim.x * blockIdx.x;
  int rowPointer = thread / ( A->getNumColumns() - colPointerMain );
  int colPointer = thread % ( A->getNumColumns() - colPointerMain ) + colPointerMain;
  
  if( colPointer > colPointerMain && colPointer < A->getNumColumns() && 
          rowPointer + processID * A->getNumRows() != colPointerMain && rowPointer < A->getNumRows() )
  {
    if( mainRow[ 0 ] != 0 )
    { 
      const Real pivot = mainRow[ 0 ];
      const Real firstElementInRow = A->getElement( rowPointer, colPointerMain );
      if( firstElementInRow != 0 )
      {
        A->setElement( rowPointer, colPointer,
                  A->getElement( rowPointer, colPointer ) - firstElementInRow * mainRow[ colPointer - colPointerMain ] / pivot );   
      }
    } else if( colPointer == colPointerMain && rowPointer == colPointerMain ) printf( "Error, pivot is zero!\n");
  }
  if( colPointer == colPointerMain && rowPointer + processID * A->getNumRows() != colPointerMain && rowPointer < A->getNumRows() 
          && colPointer < A->getNumColumns() && mainRow[ 0 ] != 0 && A->getElement( rowPointer, colPointerMain ) != 0  )
  {
    b[ rowPointer ] = b[ rowPointer ] - A->getElement( rowPointer, colPointerMain ) * mainRow[ A->getNumColumns() - colPointerMain ] / mainRow[ 0 ];
  }
}


template <typename Real >
__global__ 
void GEMDiagToResult( Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > v,
        int processID )
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if( i < A->getNumColumns() - processID*A->getNumRows() && i < A->getNumRows() )
    v[i] = v[i] / A->getElement( i, i + processID * A->getNumRows() );
}


/*********************FIRST TRY WITH COLUMNS************************************/

template <typename Real >
__global__ 
void GEMForwardPass( Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b, int rowPointer )
{
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row < A->getNumRows() && row > rowPointer )
  {
    if( A->getElement( rowPointer, rowPointer ) != 0 )
    {    
      const Real pivot = A->getElement( rowPointer, rowPointer );
      const Real firstElementInRow = A->getElement( row, rowPointer );
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


template < typename Real >
__global__
void GEMNormRows( Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b )
{
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row < A->getNumRows() )
  {
    const Real firstElementInRow = A->getElement(row,row);
    for( int i = row; i < A->getNumRows(); i++ )
    {
      A->setElement( row, i, A->getElement( row, i ) / firstElementInRow );
    }
    b[ row ] = b[ row ] / firstElementInRow;
  }
}



template < typename Real >
__global__
void GEMBackwardPass( Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b, int rowPointer )
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



template <typename Real, typename Device>
__global__ 
void showMatrix( Matrix< Real, Device, int > A)
{
  A.showMatrix();
}



/****************************FIRST TRY WITH BLOCKS******************************/

template <typename Real>
__global__
void GEMBlocks( Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b, int mainBlockPointer )
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
          const Real pivot = A->getElement( mainRowForBlock, rowPointer );
          const Real firstElementInRow = A->getElement( row, rowPointer );
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
          const Real pivot = A->getElement( mainRowForBlock, rowPointer );
          const Real firstElementInRow = A->getElement( row, rowPointer );
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

template <typename Real>
__global__
void GEMZeroing(  Matrix< Real, TNL::Devices::Cuda, int >* A,
        TNL::Containers::VectorView< Real, TNL::Devices::Cuda, int > b, int mainBlockPointer )
{
  if( threadIdx.x == 0 && blockIdx.x == 1 )
    A->showMatrix();
  int mainRowForThisRow = mainBlockPointer*blockDim.x + threadIdx.x;
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  if( row > mainRowForThisRow && row < A->getNumRows() && mainRowForThisRow < A->getNumColumns() && mainBlockPointer < blockIdx.x )
  {
    const Real pivot = A->getElement( mainRowForThisRow, mainRowForThisRow );
    const Real firstElement = A->getElement( row, mainRowForThisRow );
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

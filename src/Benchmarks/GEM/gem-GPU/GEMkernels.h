#ifdef HAVE_CUDA
//TODO: Real



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



__global__ 
void showMatrix( Matrix< double, TNL::Devices::Cuda, int > A)
{
  A.showMatrix();
}


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


#endif //HAVE_CUDA

#include "TNL/Cuda/MemoryHelpers.h"
#define DEBUG 0



#ifdef HAVE_CUDA
#include "GEMkernels.h"

template < typename Real,
        typename Index >
void calculResultVector( Matrix< Real, TNL::Devices::Cuda, Index >& matrix,
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& x )
{ 
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( matrix);
  
  int blockSize = matrix.getNumRows() > 1024 ? 1024 : matrix.getNumColumns();
  int numBlocksOnColumn = TNL::roundUpDivision( matrix.getNumRows(), 1024 );
  int numOfBlocks =  matrix.getNumRows() * numBlocksOnColumn;
    
  
  GEMDiagToResult<<< numOfBlocks, blockSize >>>( devMat,device_vector.getView(), x.getView() );
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
} 




template < typename Real,
        typename Device,
        typename Index >
bool GEM<Real, Device, Index >::GEMdevice( Array& x, const TNL::String& pivoting, int verbose )
{
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( this->A );
  TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector( this->b );
  
  // FOR PIVOTING SET VARIABLES ON DEVICE
  size_t size = sizeof(int);
  int* pivot; cudaMalloc(&pivot, size);
  int* pom = (int*)malloc(size); *pom = -1;
  
  if( verbose > 2 )
  {
    showMatrix<<< 1, 1 >>>( this->A );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");
  }
  
  
  for( int colPointer = 0; colPointer < this->A.getNumColumns(); colPointer++ )
  {
    if( verbose > 1 )
      printf( "Elimination: %d/%d\n", colPointer, this->A.getNumColumns() );
    
    if( pivoting == "yes" )
    {
      // PIVOTING
      int reduceBlockSize = (this->A.getNumColumns()-colPointer) > 1024 ? 1024 : 
        TNL::roundToMultiple( this->A.getNumColumns()-colPointer, 32 );  
      int reduceGridSize = TNL::roundUpDivision( this->A.getNumColumns()-colPointer, reduceBlockSize );
      int reduceGridSizeRound = TNL::roundToMultiple( reduceGridSize, 32 );
      //printf("%d, %d, %d\n", reduceBlockSize, reduceGridSize, reduceGridSizeRound );

      TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > outMax(reduceGridSize);
      TNL::Containers::Vector< Index, TNL::Devices::Cuda, Index > outPos(reduceGridSize);
      //outMax.setValue(0); outPos.setValue(0);

      findPivot<<< reduceGridSize, reduceBlockSize >>>( devMat, colPointer, outMax.getView(), outPos.getView() );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;

      findRowPivot<<< 1, reduceGridSizeRound >>>( outMax.getView(), outPos.getView(), pivot );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
      *pom = 0;
      cudaMemcpy( pom, pivot, size, cudaMemcpyDeviceToHost);
    }
    
    
    int blockSize = (this->A.getNumColumns()-colPointer) > 1024 ? 1024 : this->A.getNumColumns()-colPointer;
    int numBlocksOnRow = TNL::roundUpDivision( (this->A.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  this->A.getNumRows() * numBlocksOnRow;
    
    
    if( pivoting == "yes" )// && *pom != -1 && *pom != colPointer )
    {
      if( verbose > 1 )
      {
         std::cout << std::endl;
         std::cout << "Choosing element at " << *pom << "-th row as pivot with value..."  << std::endl;
         std::cout << "Swapping " << colPointer << "-th and " << *pom <<  "-th rows ... " << std::endl;
      }
      swapRows<<< numBlocksOnRow, blockSize >>>( devMat, device_vector.getView(), colPointer, numBlocksOnRow, pivot );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
    } 
    
    GEMmainKernel<<< numOfBlocks, blockSize >>>( devMat, 
                                                  device_vector.getView(), 
                                                  colPointer, 
                                                  numBlocksOnRow );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
  }
  
  cudaFree(pivot);
  free(pom);
  
  calculResultVector( this->A, device_vector, x );
  
  return true;
}

#endif // HAVE_CUDA
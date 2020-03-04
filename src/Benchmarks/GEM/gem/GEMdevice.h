#include "TNL/Cuda/MemoryHelpers.h"
#define DEBUG 0



#ifdef HAVE_CUDA
#include "GEMkernels.h"

template < typename Real,
        typename Index >
void calculateResultSeqCPU( Matrix< Real, TNL::Devices::Cuda, Index >& matrix,
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
bool GEM<Real, Device, Index >::GEMdevice( /*Matrix< Real, TNL::Devices::Cuda, Index >& this->A, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector, */
                Array& x, const TNL::String& pivoting, int verbose )
{
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( this->A );
  TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector( this->b );
  // FOR PIVOTING SET VARIABLES ON DEVICE
  size_t size = sizeof(int);
  int* pivot; cudaMalloc(&pivot, size);
  
  
  for( int colPointer = 0; colPointer < this->A.getNumColumns(); colPointer++ )
  {
    
    // PIVOTING
    int reduceBlockSize = (this->A.getNumColumns()-colPointer) > 512 ? 512 : 
      TNL::roundToMultiple( this->A.getNumColumns()-colPointer, 32 );  
    int reduceGridSize = TNL::roundUpDivision( this->A.getNumColumns()-colPointer, reduceBlockSize );
    int reduceGridSizeRound = TNL::roundToMultiple( reduceGridSize, 32 );
    
    TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > outMax(reduceGridSizeRound);
    TNL::Containers::Vector< Index, TNL::Devices::Cuda, Index > outPos(reduceGridSizeRound);
    outMax.setValue(0); outPos.setValue(0);
        
    findPivot<<< reduceGridSize, reduceBlockSize >>>( devMat, colPointer, outMax.getView(), outPos.getView() );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    findRowPivot<<< 1, reduceGridSizeRound >>>( outMax.getView(), outPos.getView(), pivot );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    int* pom = (int*)malloc(size); *pom = 0;
    cudaMemcpy( pom, pivot, size, cudaMemcpyDeviceToHost);
    //printf("%d: position of pivot: %d\n", colPointer, *pom);
    
    
    int blockSize = (this->A.getNumColumns()-colPointer) > 1024 ? 1024 : this->A.getNumColumns()-colPointer;
    int numBlocksOnRow = TNL::roundUpDivision( (this->A.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  this->A.getNumRows() * numBlocksOnRow;
    //printf( "%d number of threads, %d number of blocks\n", blockSize, numOfBlocks);
    
    if( *pom != -1 && *pom != colPointer )
    {
      swapRows<<< numBlocksOnRow, blockSize >>>( devMat, device_vector.getView(), colPointer, numBlocksOnRow, pivot );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
    } 
    
    /*printf("\n");
    showMatrix<<< 1, 1 >>>( this->A );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");*/
    
    GEMColumnUnderDiag<<< numOfBlocks, blockSize >>>( devMat, 
                                                      device_vector.getView(), 
                                                      colPointer, 
                                                      numBlocksOnRow );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    
    
    /*int blockSize1 = this->A.getNumRows() > 1024 ? 1024 : this->A.getNumRows();
    int gridSize1 = TNL::roundUpDivision( this->A.getNumRows(), blockSize1 );
    GEMZeroingMainColumn<<< gridSize1, blockSize1 >>>(devMat, 
                                                      device_vector.getView(), 
                                                      colPointer );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    printf("\n");
    showMatrix<<< 1, 1 >>>( this->A );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");
    std::cout << device_vector << std::endl;*/
  }
  
  cudaFree(pivot);
  //std::cout << device_vector << std::endl;
  
  calculateResultSeqCPU( this->A, device_vector, x );
  
  return true;
}

#endif // HAVE_CUDA
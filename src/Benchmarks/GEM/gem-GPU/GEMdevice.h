#include "GEMkernels.h"
#include "TNL/Cuda/MemoryHelpers.h"
#define DEBUG 0



#ifdef HAVE_CUDA

template < typename Real,
        typename Index >
void calculateResultSeqCPU( Matrix< Real, TNL::Devices::Cuda, Index >& matrixDev,
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& result_vector_dev )
{ 
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( matrixDev);
  
  int blockSize = matrixDev.getNumRows() > 1024 ? 1024 : matrixDev.getNumColumns();
  int numBlocksOnColumn = TNL::roundUpDivision( matrixDev.getNumRows(), 1024 );
  int numOfBlocks =  matrixDev.getNumRows() * numBlocksOnColumn;
    
  
  GEMDiagToResult<<< numOfBlocks, blockSize >>>( devMat,device_vector.getView(),result_vector_dev.getView() );
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
} 








template < typename Real,
        typename Index >
void GEMdevice( Matrix< Real, TNL::Devices::Cuda, Index >& matrixDev, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& result_vector_dev )
{
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( matrixDev);
    
  // FOR PIVOTING SET VARIABLES ON DEVICE
  size_t size = sizeof(int);
  int* pivot; cudaMalloc(&pivot, size);
  
  
  for( int colPointer = 0; colPointer < matrixDev.getNumColumns(); colPointer++ )
  {
    int reduceBlockSize = (matrixDev.getNumColumns()-colPointer) > 512 ? 512 : 
      TNL::roundToMultiple( matrixDev.getNumColumns()-colPointer, 32 );  
    int reduceGridSize = TNL::roundUpDivision( matrixDev.getNumColumns()-colPointer, reduceBlockSize );
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
    
    
    int blockSize = (matrixDev.getNumColumns()-colPointer) > 1024 ? 1024 : matrixDev.getNumColumns();
    int numBlocksOnRow = TNL::roundUpDivision( (matrixDev.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  matrixDev.getNumRows() * numBlocksOnRow;
    //printf( "%d number of threads, %d number of blocks\n", blockSize, numOfBlocks);
    
    if( *pom != -1 && *pom != colPointer )
    {
      swapRows<<< numBlocksOnRow, blockSize >>>( devMat, device_vector.getView(), colPointer, numBlocksOnRow, pivot );
    }
    
    /*showMatrix<<< 1, 1 >>>( matrixDev );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;*/
    
    /*int* pom = (int*)malloc(size); *pom = 0;
    cudaMemcpy(pivot, pom, size, cudaMemcpyHostToDevice);
    findPivot<<< numBlocksOnRow, 1024 >>>( devMat, colPointer, numBlocksOnRow  );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    findRowPivot<<< numBlocksOnRow, blockSize >>>( devMat, colPointer, numBlocksOnRow, d_max, pivot );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    
    cudaMemcpy( pom, pivot, size, cudaMemcpyDeviceToHost);
    //printf("%d: position of pivot: %d\n", colPointer, *pom);
    if( *pom != -1 && *pom != colPointer )
    {
      swapRows<<< numBlocksOnRow, blockSize >>>( devMat, device_vector.getView(), colPointer, numBlocksOnRow, pivot );
    }*/
    
    
    /*cudaMemcpy( pom, d_max, size, cudaMemcpyDeviceToHost);
    printf("%d\n", *pom );*/
    
    
    
    GEMColumnUnderDiag<<< numOfBlocks, blockSize >>>( devMat, 
                                                      device_vector.getView(), 
                                                      colPointer, 
                                                      numBlocksOnRow );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    /*printf("\n");
    showMatrix<<< 1, 1 >>>( matrixDev );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");*/
  }
  
  cudaFree(pivot);
  //std::cout << device_vector << std::endl;
  
  calculateResultSeqCPU( matrixDev, device_vector, result_vector_dev );
  
  
}

#endif // HAVE_CUDA
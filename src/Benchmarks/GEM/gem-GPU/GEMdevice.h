#include "GEMkernels.h"
#include "TNL/Cuda/MemoryHelpers.h"
#define DEBUG 0
#ifdef HAVE_CUDA
template < typename Real,
        typename Index >
void GEMdevice( Matrix< Real, TNL::Devices::Cuda, Index >& matrixDev, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector, 
                TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& result_vector_dev )
{
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat = TNL::Cuda::passToDevice( matrixDev);
  int blockSize = 3;
  int numOfBlocks = TNL::roundUpDivision( matrixDev.getNumRows(), blockSize );
  printf( "%d number of threads, %d number of blocks\n", blockSize, numOfBlocks);
  
  for( int mainBlockId = 0; mainBlockId < numOfBlocks; mainBlockId++ )
  {
    GEMBlocks<<< numOfBlocks, blockSize >>>( devMat, device_vector.getView(), mainBlockId );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    GEMZeroing<<< numOfBlocks, blockSize >>>( devMat, device_vector.getView(), mainBlockId );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
  }
  
#if DEBUG
  showMatrix<<< 1, 1 >>>( matrixDev );
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;

  std::cout << device_vector << "\n" << std::endl;
#endif
  
}
#endif

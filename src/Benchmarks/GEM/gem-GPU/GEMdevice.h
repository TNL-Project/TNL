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
  Matrix< double, TNL::Devices::Host, int> matrix( matrixDev.getNumRows(), matrixDev.getNumColumns() );
  matrix = matrixDev;
  TNL::Containers::Vector< double, TNL::Devices::Host, int > host_vec( matrixDev.getNumRows() ) ,result( matrixDev.getNumRows() );
  host_vec = device_vector;
  
  int n = matrix.getNumRows();
  for( int k = n - 1; k >= 0; k-- )
   {
      //if( k % 10 == 0 )
      //   std::cout << "Substitution: " << k << "/" << n << std::endl;
      result[ k ] = host_vec[ k ] / matrix.getElement(k,k);
      for( int j = k + 1; j < n; j++ )
         result[ k ] -= result[ j ] * matrix.getElement( k, j )/matrix.getElement(k,k);
   }
  result_vector_dev = result;
} 








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
#if DEBUG
    showMatrix<<< 1, 1 >>>( matrixDev );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;

    std::cout << device_vector << "\n" << std::endl;
#endif
    GEMZeroing<<< numOfBlocks, blockSize >>>( devMat, device_vector.getView(), mainBlockId );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#if DEBUG
    showMatrix<<< 1, 1 >>>( matrixDev );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;

    std::cout << device_vector << "\n" << std::endl;
#endif
  }
  
#if DEBUG
  showMatrix<<< 1, 1 >>>( matrixDev );
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;

  std::cout << device_vector << "\n" << std::endl;
#endif
  
  calculateResultSeqCPU( matrixDev, device_vector, result_vector_dev );
  
  
}

#endif // HAVE_CUDA
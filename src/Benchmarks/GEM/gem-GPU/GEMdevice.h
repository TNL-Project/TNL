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
    
  // FOR PIVOTING SET VARIABLES ON DEVICE
  size_t size = sizeof(int);
  int* d_max; cudaMalloc(&d_max, size);
  int* pivot; cudaMalloc(&pivot, size);
  
  
  for( int colPointer = 0; colPointer < matrixDev.getNumColumns(); colPointer++ )
  {
    int blockSize = (matrixDev.getNumColumns()-colPointer) > 1024 ? 1024 : matrixDev.getNumColumns();
    int numBlocksOnRow = TNL::roundUpDivision( (matrixDev.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  matrixDev.getNumRows() * numBlocksOnRow;
    //printf( "%d number of threads, %d number of blocks\n", blockSize, numOfBlocks);
    
    
    /*showMatrix<<< 1, 1 >>>( matrixDev );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;*/
    
    int* pom = (int*)malloc(size); *pom = 0;
    cudaMemcpy(d_max, pom, size, cudaMemcpyHostToDevice);
    cudaMemcpy(pivot, pom, size, cudaMemcpyHostToDevice);
    findPivot<<< numBlocksOnRow, blockSize >>>( devMat, colPointer, numBlocksOnRow, d_max );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    findRowPivot<<< numBlocksOnRow, blockSize >>>( devMat, colPointer, numBlocksOnRow, d_max, pivot );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    
    
    int* pom1 = (int*)malloc(size); *pom1 = 0;
    cudaMemcpy( pom1, d_max, size, cudaMemcpyDeviceToHost);    
    cudaMemcpy( pom, pivot, size, cudaMemcpyDeviceToHost);
    //printf("%d: position of pivot: %d, max %d\n", colPointer, *pom, *pom1);
    if( *pom != -1 && *pom != colPointer )
    {
      swapRows<<< numBlocksOnRow, blockSize >>>( devMat, device_vector.getView(), colPointer, numBlocksOnRow, pivot );
    }
    
    
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
  
  cudaFree(d_max);
  cudaFree(pivot);
  //std::cout << device_vector << std::endl;
  
  calculateResultSeqCPU( matrixDev, device_vector, result_vector_dev );
  
  
}

#endif // HAVE_CUDA
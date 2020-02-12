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
    
  
  for( int colPointer = 0; colPointer < matrixDev.getNumColumns(); colPointer++ )
  {
    int blockSize = (matrixDev.getNumColumns()-colPointer) > 1024 ? 1024 : matrixDev.getNumColumns();
    int numBlocksOnRow = TNL::roundUpDivision( (matrixDev.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  matrixDev.getNumRows() * numBlocksOnRow;
    printf( "%d number of threads, %d number of blocks\n", blockSize, numOfBlocks);
    GEMColumnUnderDiag<<< numOfBlocks, blockSize >>>( devMat, 
                                                      device_vector.getView(), 
                                                      colPointer, 
                                                      numBlocksOnRow );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");
    showMatrix<<< 1, 1 >>>( matrixDev );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");
  }
  std::cout << device_vector << std::endl;
  
  calculateResultSeqCPU( matrixDev, device_vector, result_vector_dev );
  
  
}

#endif // HAVE_CUDA
#define DEBUG 0



#ifdef HAVE_CUDA
//#include "GEMkernels.h"
#ifdef HAVE_MPI
#include "TNL/Communicators/MpiCommunicator.h"
#include <mpi.h>
#endif

template < typename Real,
        typename Index >
void calculResultVector( Matrix< Real, TNL::Devices::Cuda, Index >& matrix,
        TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector, 
        TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& x, 
        int processID, int numOfProcesses )
{ 
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat;// = TNL::Cuda::passToDevice( this->A );
  cudaMalloc( ( void** ) &devMat, ( size_t ) sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ) );
  cudaMemcpy( ( void* ) devMat,( void* ) &matrix, sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ), cudaMemcpyHostToDevice );
  TNL_CHECK_CUDA_DEVICE;
  
  int blockSize = matrix.getNumRows() > 1024 ? 1024 : matrix.getNumRows();
  int numBlocksOnColumn = TNL::roundUpDivision( matrix.getNumRows(), 1024 );
  int numOfBlocks =  matrix.getNumRows() * numBlocksOnColumn;
  
  
  GEMDiagToResult<<< numOfBlocks, blockSize >>>( devMat,device_vector.getView(), processID );
  cudaDeviceSynchronize();
  TNL_CHECK_CUDA_DEVICE;
#ifdef HAVE_MPI
  if( processID != 0 )
  {
    TNL::Containers::Vector< Real, TNL::Devices::Host, Index > host_vector;
    host_vector = device_vector;
    Real* data = host_vector.getData();
    
    if(processID*device_vector.getSize() < x.getSize() ){
      TNL::Communicators::MpiCommunicator::ISend( data, device_vector.getSize(), 0, 0 );
    }
      
  }else{
    for( int j = 0; j < device_vector.getSize(); j++ )
      x.setElement( j, device_vector.getElement( j ) );
    for( int i = 1; i < numOfProcesses && i*device_vector.getSize() < x.getSize(); i++ )
    {
      Real* data;
      data = new Real[ device_vector.getSize() ];
      TNL::Communicators::MpiCommunicator::Recv( data, device_vector.getSize(), i, 0 );
      
      for( int j = 0; j < device_vector.getSize() && i * device_vector.getSize() + j < x.getSize(); j++ )
        x.setElement( i * device_vector.getSize() + j, data[ j ] );
      delete []data;
    }
  }
  //std::cout << processID << ": " << x << std::endl;
#endif
   cudaFree( ( void* ) devMat );
   TNL_CHECK_CUDA_DEVICE;
} 




template < typename Real,
        typename Device,
        typename Index >
bool GEM<Real, Device, Index >::GEMdevice( Array& x, const TNL::String& pivoting, int verbose )
{
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat;// = TNL::Cuda::passToDevice( this->A );
  cudaMalloc( ( void** ) &devMat, ( size_t ) sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ) );
  cudaMemcpy( ( void* ) devMat,( void* ) &this->A, sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ), cudaMemcpyHostToDevice );
  TNL_CHECK_CUDA_DEVICE;
  TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector( this->b );
  
  // FOR PIVOTING SET VARIABLES ON DEVICE
  int* pivot; cudaMalloc(&pivot, sizeof(int));
  int* pom = (int*)malloc(sizeof(int));// *pom = -1;
  
  if( verbose > 2 )
  {
    showMatrix<<< 1, 1 >>>( this->A );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");
  }
  int processID=0;
  int numOfProcesses=1;
  
#ifdef HAVE_MPI
  MPI_Comm_rank( MPI_COMM_WORLD, &processID );
  MPI_Comm_size( MPI_COMM_WORLD, &numOfProcesses );
  Index colPointerMain = 0;
#endif
  
  while( colPointerMain < x.getSize() ){
#ifdef HAVE_MPI
    TNL::Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
    Index colPointer = colPointerMain-colPointerMain/this->A.getNumRows() * this->A.getNumRows();
    Real* mainRow;
    int size = this->A.getNumColumns() - colPointerMain + 1;
    mainRow = new Real[size];
    
    if( colPointerMain/this->A.getNumRows() == processID ){
      this->A.getRow( colPointer, colPointerMain, mainRow, size );
      mainRow[ size-1 ] = this->b.getElement( colPointer );
      
      for( int i = 0; i < numOfProcesses; i++ )
        if( i != processID ){
          TNL::Communicators::MpiCommunicator::ISend( mainRow, size, i, 0 );
        }
    } else {
      TNL::Communicators::MpiCommunicator::Recv( mainRow, size, colPointerMain/this->A.getNumRows(), 0 );
      if( verbose > 2 )
      {
        printf( "%d: [", processID);
        for( int i = 0; i < size; i++ )
          printf( "%.2f ", mainRow[ i ] );
        printf("]\n");
      }
      
    }
    TNL::Containers::Vector< Real, TNL::Devices::Host, Index > mainRowVecHost( mainRow, size );
    TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > mainRowVec;
    mainRowVec = mainRowVecHost;
    delete []mainRow;
#else
    TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > mainRowVec;
    //printf("%d: colPointer = %d, colPointerMain = %d\n", processID, colPointer, colPointerMain );
    this->A.getRow(colPointer, colPointerMain, mainRowVec );
#endif
    /*std::cout << processID << ": " << std::endl;
    std::cout << mainRowVec << std::endl;*/
    
    if( verbose > 1 )
      printf( "Elimination: %d/%d\n", colPointer, this->A.getNumColumns() );
    
    /*if( pivoting == "yes" )
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
      cudaMemcpy( pom, pivot, sizeof(int), cudaMemcpyDeviceToHost);
    }*/
    
    int blockSize = (this->A.getNumColumns()-colPointer) > 1024 ? 1024 : this->A.getNumColumns()-colPointer;
    int numBlocksOnRow = TNL::roundUpDivision( (this->A.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  this->A.getNumRows() * numBlocksOnRow;
    
    
    /*if( pivoting == "yes" )// && *pom != -1 && *pom != colPointer )
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
    } */
    
    GEMmainKernel<<< numOfBlocks, blockSize >>>( devMat, 
            device_vector.getView(),
            mainRowVec.getView(),
            colPointer, 
            colPointerMain,
            numBlocksOnRow,
            processID,
            numOfProcesses );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    if( verbose > 2 )
    {
      showMatrix<<< 1, 1 >>>( this->A );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
      std::cout << this->b << std::endl;
      printf("\n");
    }
    //TNL::Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
    colPointerMain++;
  }
  
  cudaFree(pivot);
  free(pom);
  cudaFree( devMat );
  TNL_CHECK_CUDA_DEVICE;
  
  calculResultVector( this->A, device_vector, x, processID, numOfProcesses );
  
  return true;
}

#endif // HAVE_CUDA
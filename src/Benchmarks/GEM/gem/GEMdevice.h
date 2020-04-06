#define DEBUG 0



#ifdef HAVE_CUDA
#include "GEMkernels.h"


#ifdef HAVE_MPI
#include "TNL/Communicators/MpiCommunicator.h"
#include "TNL/Communicators/MPITypeResolver.h"
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
#endif
  Index colPointerMain = 0;
  
  while( colPointerMain < x.getSize() ){
#ifdef HAVE_MPI
    TNL::Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
    Index colPointer = colPointerMain-(Index)(colPointerMain/this->A.getNumRows()) * this->A.getNumRows();
#ifdef HAVE_MPI
    if( processID == 0 )
    {
      showMatrix<<< 1, 1 >>>( this->A );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if( processID == 1 )
    {
      showMatrix<<< 1, 1 >>>( this->A );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if( processID == 2 )
    {
      showMatrix<<< 1, 1 >>>( this->A );
      cudaDeviceSynchronize();
      TNL_CHECK_CUDA_DEVICE;
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    
    // main row vector for computation (pivoting, non-pivoting)
    TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > mainRowVec;
    
    
    int blockSize = (this->A.getNumColumns()-colPointer) > 1024 ? 1024 : this->A.getNumColumns()-colPointer;
    int numBlocksOnRow = TNL::roundUpDivision( (this->A.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  this->A.getNumRows() * numBlocksOnRow;
    
    if( pivoting == "yes" )
    {
      // PIVOTING
      Index fromRow = 0;
      if( processID < (Index)(colPointerMain/this->A.getNumRows()) )
        fromRow = this->A.getNumRows();
      else if( processID == (Index)(colPointerMain/this->A.getNumRows()) )
        fromRow = colPointer;
      
      TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > outMax(1);
      TNL::Containers::Vector< Index, TNL::Devices::Cuda, Index > outPos(1);
      outMax.setValue(0); outPos.setValue(-1);
      
      if( fromRow != this->A.getNumRows() )
      {
        int reduceBlockSize = (this->A.getNumRows()-fromRow) > 1024 ? 1024 : 
          TNL::roundToMultiple( this->A.getNumRows()-fromRow, 32 );  
        int reduceGridSize = TNL::roundUpDivision( this->A.getNumRows()-fromRow, reduceBlockSize );
        int reduceGridSizeRound = TNL::roundToMultiple( reduceGridSize, 32 );
        
        printf("%d,%d: reduceBlockSize = %d, reduceGridSize = %d, reduceGridSizeRound = %d %d\n",
                colPointerMain, processID, reduceBlockSize, reduceGridSize, reduceGridSizeRound, fromRow );
        
        outMax.setSize( reduceGridSize );
        outPos.setSize( reduceGridSize );
        //outMax.setValue(0); outPos.setValue(0);
        
        findPivot<<< reduceGridSize, reduceBlockSize >>>( devMat, fromRow, colPointerMain, outMax.getView(), outPos.getView() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        findRowPivot<<< 1, reduceGridSizeRound >>>( outMax.getView(), outPos.getView(), pivot );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE; 
        //*pom = 0;
        //cudaMemcpy( pom, pivot, sizeof(int), cudaMemcpyDeviceToHost);
      }
      //std::cout << processID << ": " << outMax << outPos << std::endl;
      
      Real *data, *recvData;
      data = new Real[2];
      recvData = new Real[2*numOfProcesses];
      data[0] = outMax.getElement(0); data[1] = outPos.getElement(0);
      MPI_Barrier( MPI_COMM_WORLD );
      
      MPI_Allgather( data, 2, TNL::Communicators::MPITypeResolver< Real >::getType(),
              recvData, 2, TNL::Communicators::MPITypeResolver< Real >::getType(), MPI_COMM_WORLD);
      
      /*MPI_Barrier( MPI_COMM_WORLD );
      if( processID == 0 )
      {
        printf( "%d:", processID );
        for( int i = 0; i < 2*numOfProcesses; i++ )
          printf( "%.2f ", recvData[i] );
        printf("\n"); 
      }
      MPI_Barrier( MPI_COMM_WORLD );
      if( processID == 1 )
      {
        printf( "%d:", processID );
        for( int i = 0; i < 2*numOfProcesses; i++ )
          printf( "%.2f ", recvData[i] );
        printf("\n"); 
      }*/
      
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > outMaxHost( numOfProcesses );
      TNL::Containers::Vector< Index, TNL::Devices::Host, Index > outPosHost( numOfProcesses );
           
      for( int i = 0; i < numOfProcesses; i++ )
      {
        outMaxHost[ i ] = (Real)recvData[ 2*i ];
        outPosHost[ i ] = (Index)recvData[ 2*i+1 ];
      }
      delete []data;
      delete []recvData;
      
      
      /*if( processID == 0 )
      {
        printf( "%d:", processID );
        std::cout << outMaxHost << std::endl;
        std::cout << outPosHost << std::endl;
      }
      MPI_Barrier( MPI_COMM_WORLD );
      if( processID == 1 )
      {
        printf( "%d: ", processID );
        std::cout << outMaxHost << std::endl;
        std::cout << outPosHost << std::endl;
      }*/
      
      Index ProcessMax = -1;
      Index Possition = -1;
      Real Maximum = 0;
      
      for( int i = 0; i < numOfProcesses; i++ )
      {
        if( outPosHost[ i ] != -1 )
          if( Maximum < outMaxHost[ i ] )
          {
            Maximum = outMaxHost[ i ];
            ProcessMax = i;
            Possition = outPosHost[ i ];
          }
      }
      
      /*if( processID == 0 )
      {
        printf( "%d: ", processID );
        std::cout << ProcessMax << " " << Maximum << " " << Possition << std::endl; 
      }
      MPI_Barrier( MPI_COMM_WORLD );
      if( processID == 1 )
      {
        printf( "%d: ", processID );
        std::cout << ProcessMax << " " << Maximum << " " << Possition << std::endl; 
      }*/
      Real* mainRow;
      int size = this->A.getNumColumns() - colPointerMain + 1;
      mainRow = new Real[size];
      
      if( ProcessMax != colPointerMain/this->A.getNumRows() )
      {
        if( processID == ProcessMax )
        {
          this->A.getRow( Possition, colPointerMain, mainRow, size );
          mainRow[ size-1 ] = this->b.getElement( Possition );
        }
      } else {
        if( colPointerMain/this->A.getNumRows() == processID ){
          if( Possition != colPointer )
          {
            if( verbose > 1 )
            {
              std::cout << std::endl;
              std::cout << "Choosing element at " << Possition << "-th row as pivot with value..."  << std::endl;
              std::cout << "Swapping " << colPointer << "-th and " << Possition <<  "-th rows ... " << std::endl;
            }
            swapRows<<< numBlocksOnRow, blockSize >>>( devMat, device_vector.getView(), colPointer, numBlocksOnRow, Possition );
            cudaDeviceSynchronize();
            TNL_CHECK_CUDA_DEVICE;
          }
          this->A.getRow( colPointer, colPointerMain, mainRow, size );
          mainRow[ size-1 ] = this->b.getElement( colPointer );
        } 
      }
      TNL::Communicators::MpiCommunicator::Bcast( mainRow, size, colPointerMain/this->A.getNumRows(), MPI_COMM_WORLD);
      
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > mainRowVecHost( mainRow, size );
      mainRowVec = mainRowVecHost;
      delete []mainRow;
      
      if( ProcessMax != colPointerMain/this->A.getNumRows() )
      {
        Real *mainRowSwap;
        mainRowSwap = new Real[size];
        
        if( processID == ProcessMax )
        {
          TNL::Communicators::MpiCommunicator::Recv( mainRowSwap, size, colPointerMain/this->A.getNumRows(), 0 );
          printf( "%d: ", processID );
          for( int i = 0; i < size; i++ )
            printf( "%.2f ", mainRowSwap[ i ] );
          printf( "\n" );
          this->A.setRow( Possition, colPointerMain, mainRowSwap, size );
          this->b.setElement( Possition, mainRowSwap[ size-1 ] );
        }
        else if( processID == colPointerMain/this->A.getNumRows() )
        {
          this->A.getRow( colPointer, colPointerMain, mainRowSwap, size );
          mainRowSwap[ size-1 ] = this->b.getElement( colPointer );
          
          TNL::Communicators::MpiCommunicator::ISend( mainRowSwap, size, ProcessMax, 0 );
          printf( "%d: ", processID );
          for( int i = 0; i < size; i++ )
            printf( "%.2f ", mainRow[ i ] );
          printf( "\n" );
          this->A.setRow( colPointer, colPointerMain, mainRow, size );
          this->b.setElement( colPointer, mainRow[ size-1 ] );
        }        
        delete []mainRowSwap;
      }
      #ifdef HAVE_MPI
       if( processID == 0 )
       {
       showMatrix<<< 1, 1 >>>( this->A );
       cudaDeviceSynchronize();
       TNL_CHECK_CUDA_DEVICE;
       }
       MPI_Barrier(MPI_COMM_WORLD);
       if( processID == 1 )
       {
       showMatrix<<< 1, 1 >>>( this->A );
       cudaDeviceSynchronize();
       TNL_CHECK_CUDA_DEVICE;
       }
       MPI_Barrier(MPI_COMM_WORLD);
       if( processID == 2 )
       {
       showMatrix<<< 1, 1 >>>( this->A );
       cudaDeviceSynchronize();
       TNL_CHECK_CUDA_DEVICE;
       }
       MPI_Barrier(MPI_COMM_WORLD);
       if( processID == 0 )
       std::cout << this->b << std::endl;
       MPI_Barrier(MPI_COMM_WORLD);
       if( processID == 1 )
       std::cout << this->b << std::endl;
       MPI_Barrier(MPI_COMM_WORLD);
       if( processID == 2 )
       std::cout << this->b << std::endl;
       #endif
    }
    else 
    {
#ifdef HAVE_MPI
      Real* mainRow;
      int size = this->A.getNumColumns() - colPointerMain + 1;
      mainRow = new Real[size];
      
      if( colPointerMain/this->A.getNumRows() == processID ){
        this->A.getRow( colPointer, colPointerMain, mainRow, size );
        mainRow[ size-1 ] = this->b.getElement( colPointer );
        
        /*for( int i = 0; i < numOfProcesses; i++ )
         if( i != processID ){
         TNL::Communicators::MpiCommunicator::ISend( mainRow, size, i, 0 );
         }*/
      } //else {
      //TNL::Communicators::MpiCommunicator::Recv( mainRow, size, colPointerMain/this->A.getNumRows(), 0 );
      TNL::Communicators::MpiCommunicator::Bcast( mainRow, size, colPointerMain/this->A.getNumRows(), MPI_COMM_WORLD);
      if( verbose > 2 )
      {
        printf( "%d: [", processID);
        for( int i = 0; i < size; i++ )
          printf( "%.2f ", mainRow[ i ] );
        printf("]\n");
      }
      
      //}
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > mainRowVecHost( mainRow, size );
      mainRowVec = mainRowVecHost;
      delete []mainRow;
#else
      //printf("%d: colPointer = %d, colPointerMain = %d\n", processID, colPointer, colPointerMain );
      this->A.getRow(colPointer, colPointerMain, mainRowVec );
#endif
      
    }
    /*std::cout << processID << ": " << std::endl;
     std::cout << mainRowVec << std::endl;*/
    
    if( verbose > 1 )
      printf( "Elimination: %d/%d\n", colPointer, this->A.getNumColumns() );
    
    
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
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
#else
  x = device_vector;
#endif
  cudaFree( ( void* ) devMat );
  TNL_CHECK_CUDA_DEVICE;
} 




template < typename Real,
        typename Device,
        typename Index >
bool GEM<Real, Device, Index >::GEMdevice( Array& x, const TNL::String& pivoting, int verbose )
{
  // Copy matrix A and vector b to GPU 
  Matrix< Real, TNL::Devices::Cuda, Index >* devMat;
  cudaMalloc( ( void** ) &devMat, ( size_t ) sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ) );
  cudaMemcpy( ( void* ) devMat,( void* ) &this->A, sizeof( Matrix< Real, TNL::Devices::Cuda, Index > ), cudaMemcpyHostToDevice );
  TNL_CHECK_CUDA_DEVICE;
  TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index >& device_vector( this->b );
  
  // FOR PIVOTING SET VARIABLES ON DEVICE
  int* pivot; cudaMalloc(&pivot, sizeof(int));
    
  // Initialise MPI variables even without MPI
  int processID=0;
  int numOfProcesses=1;
  
#ifdef HAVE_MPI
  MPI_Comm_rank( MPI_COMM_WORLD, &processID );
  MPI_Comm_size( MPI_COMM_WORLD, &numOfProcesses );
#endif
  
  if( verbose > 2 )
  {
#ifdef HAVE_MPI
    for( int i = 0; i < numOfProcesses; i++ )
    {
      if( processID == i )
      {
        showMatrix<<< 1, 1 >>>( this->A );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    for( int i = 0; i < numOfProcesses; i++ )
    {
      if( i == processID )
        std::cout << device_vector;
      MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << std::endl;
#else
    showMatrix<<< 1, 1 >>>( this->A );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    printf("\n");
#endif
  }
  
  // Main pointer to row, over all parts of matrices, colPointerMain in (0 - number of rows)
  Index colPointerMain = 0;
  
  // Main cycle for all rows across all MPI parts, vector x is the only one with full size on MPI, or use A.getNumColumns() for rectangular matrices.
  while( colPointerMain < x.getSize() ){
#ifdef HAVE_MPI
    TNL::Communicators::MpiCommunicator::Barrier( MPI_COMM_WORLD );
#endif
    // Pointer to rows in main process that has colPointerMain.
    Index colPointer = colPointerMain-TNL::floor(colPointerMain/this->A.getNumRows()) * this->A.getNumRows();

    
    // main row vector for computation (pivoting, non-pivoting)
    TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > mainRowVec( this->A.getNumColumns() - colPointerMain + 1 );
    
    // Setting number of threads and blocks for main kernel and for pivoting swapping kernel
    int blockSize = (this->A.getNumColumns()-colPointer) > 1024 ? 1024 : this->A.getNumColumns()-colPointer;
    int numBlocksOnRow = TNL::roundUpDivision( (this->A.getNumColumns()-colPointer), 1024 );
    int numOfBlocks =  this->A.getNumRows() * numBlocksOnRow;
    
    
    if( pivoting == "yes" )
    {
      // PIVOTING
      // fromRow saves info for each process saying from which row to start looking for pivot
      Index fromRow = 0;
      if( processID < TNL::floor(colPointerMain/this->A.getNumRows()) )
        fromRow = this->A.getNumRows();
      else if( processID == TNL::floor(colPointerMain/this->A.getNumRows()) )
        fromRow = colPointer;
      
      // Inicialising vectors for maximum and position (vectors important for multiple blocks, otherwise its vector with length 1)
      TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > outMax(1);
      TNL::Containers::Vector< Index, TNL::Devices::Cuda, Index > outPos(1);
      outMax.setValue(0); outPos.setValue(-1);
      
      
      // those blocks that have rows to look for pivot in should start looking
      if( fromRow != this->A.getNumRows() )
      {
        // setting size for kernel of pivoting
        int reduceBlockSize = (this->A.getNumRows()-fromRow) > 1024 ? 1024 : 
          TNL::roundToMultiple( this->A.getNumRows()-fromRow, 32 );  
        int reduceGridSize = TNL::roundUpDivision( this->A.getNumRows()-fromRow, reduceBlockSize );
        int reduceGridSizeRound = TNL::roundToMultiple( reduceGridSize, 32 );
        
        // resizing outMax and outPos for the kernel of pivoting
        outMax.setSize( reduceGridSize );
        outPos.setSize( reduceGridSize );
        
        // two main pivoting kernels executes and saves max and position into 0-th element of vectors outMax and outPos
        findPivot<<< reduceGridSize, reduceBlockSize >>>( devMat, fromRow, colPointerMain, outMax.getView(), outPos.getView() );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
        
        findRowPivot<<< 1, reduceGridSizeRound >>>( outMax.getView(), outPos.getView(), pivot );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
      }
      // Now each process has info about maximum in the "active" part of matrix that they calculate with
      
      // Preparing dynamic arrays for MPI send and recv resp. AllGather.
      // data stores information to send from each process
      // recvData stores information that is received
#ifdef HAVE_MPI
      Real *data, *recvData;
      data = new Real[2];
      recvData = new Real[2*numOfProcesses];
      data[0] = outMax.getElement(0); data[1] = outPos.getElement(0);
      MPI_Barrier( MPI_COMM_WORLD );
      
      MPI_Allgather( data, 2, TNL::Communicators::MPITypeResolver< Real >::getType(),
              recvData, 2, TNL::Communicators::MPITypeResolver< Real >::getType(), MPI_COMM_WORLD);
#endif      
           
      // Initialising maximum and possition and process that has the overall maximum ( ?: for non-MPI program)
      Index ProcessMax = numOfProcesses != 1 ? -1 : 0;
      Index Possition = numOfProcesses != 1 ? 0 : outPos.getElement(0);
      Real Maximum = numOfProcesses != 1 ? 0 : outMax.getElement(0);
      
#ifdef HAVE_MPI
      // clasic maximum finding + storing processMax and possition
      for( int i = 0; i < 2*numOfProcesses; i = i+2 )
      {
        if( recvData[ i+1 ] != -1 )
          if( Maximum < recvData[ i ] )
          {
            Maximum = recvData[ i ];
            ProcessMax = TNL::floor( i/2 );
            Possition = recvData[ i+1 ];
          }
      }
      if( verbose > 1 && processID == 0 )
        printf("%d: max = %.2f, possition = %d, process = %d\n", colPointerMain, Maximum, Possition, ProcessMax );
      // All processes has the info in Maximum, ProcesMax and Possition. So deleting arrays.
      delete []data;
      delete []recvData;
#endif
      // Clasic Maximum == 0 then we occured zero pivot so ending this calculation
      if( Maximum == 0 )
      {
        std::cout << "Ooops zero pivot occured in " << colPointerMain << "-th step." << std::endl;
        return false;
      }
      
      // Now when every process has the ProcessMax of pivoting row across all processes
      // we can send pivoting row to all processes from ProcessMax
      // mainRow stores pivoting row
      Real* mainRow;
      int size = this->A.getNumColumns() - colPointerMain + 1;
      mainRow = new Real[size];
      
      
      // If ProcessMax isn't the main process that contains colPointerMain then ProcessMax sets mainRow itself.
      // else means that ProcessMax is in the main process that contains colPointerMain, in this case we need to do normal pivoting
      // so it swaps rows and fills mainRow normally.
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
      
      // Broad casting the pivoting row to all processes
#ifdef HAVE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
      TNL::Communicators::MpiCommunicator::Bcast( mainRow, size, ProcessMax, MPI_COMM_WORLD);
      
      if( verbose > 1 )
      {
        printf( "%d: [", processID);
        for( int i = 0; i < size; i++ )
          printf( "%.2f ", mainRow[ i ] );
        printf("]\n");
      }
      
      
      // Onec more if the ProcessMax filled the mainRow, then the ProcessMax needs to switch this pivoting row with main process.
      // mainRowSwap is the colPointer of process colPointerMain/this->A.getNumRows()
      if( ProcessMax != colPointerMain/this->A.getNumRows() )
      {
        Real *mainRowSwap;
        mainRowSwap = new Real[size];
        
        
        if( processID == ProcessMax )
        {
          TNL::Communicators::MpiCommunicator::Recv( mainRowSwap, size, colPointerMain/this->A.getNumRows(), 0 );
          this->A.setRow( Possition, colPointerMain, mainRowSwap, size );
          this->b.setElement( Possition, mainRowSwap[ size-1 ] );
        }
        else if( processID == colPointerMain/this->A.getNumRows() )
        {
          this->A.getRow( colPointer, colPointerMain, mainRowSwap, size );
          mainRowSwap[ size-1 ] = this->b.getElement( colPointer );
          
          TNL::Communicators::MpiCommunicator::ISend( mainRowSwap, size, ProcessMax, 0 );
          this->A.setRow( colPointer, colPointerMain, mainRow, size );
          this->b.setElement( colPointer, mainRow[ size-1 ] );
        }    
        delete []mainRowSwap;
      }
#endif
      // Main kernel works with vector as a main row, so all processes has to set mainRowVec.
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > mainRowVecHost( mainRow, size );
      mainRowVec = mainRowVecHost;
      delete []mainRow; 
    }
    else // without pivoting
    {
#ifdef HAVE_MPI
      Real* mainRow;
      int size = this->A.getNumColumns() - colPointerMain + 1;
      mainRow = new Real[size];
      
      if( colPointerMain/this->A.getNumRows() == processID ){
        this->A.getRow( colPointer, colPointerMain, mainRow, size );
        mainRow[ size-1 ] = this->b.getElement( colPointer );
      } 
      
      TNL::Communicators::MpiCommunicator::Bcast( mainRow, size, colPointerMain/this->A.getNumRows(), MPI_COMM_WORLD);
      
      if( verbose > 2 )
      {
        printf( "%d: [", processID);
        for( int i = 0; i < size; i++ )
          printf( "%.2f ", mainRow[ i ] );
        printf("]\n");
      }
      
      TNL::Containers::Vector< Real, TNL::Devices::Host, Index > mainRowVecHost( mainRow, size );
      mainRowVec = mainRowVecHost;
      delete []mainRow;
#else
      this->A.getRow(colPointer, colPointerMain, mainRowVec );
      mainRowVec.setElement( mainRowVec.getSize() - 1, this->b.getElement( colPointer ) );
#endif
    }
    if( verbose > 1 )
    {
#ifdef HAVE_MPI
      for( int i = 0; i < numOfProcesses; i++ )
      {
        if( processID == i )
        {
          std::cout << mainRowVec << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
#else
      std::cout << mainRowVec << std::endl;
#endif
    }
    if( verbose > 1 )
      printf( "Elimination: %d/%d\n", colPointerMain, this->A.getNumColumns() );
    //std::cout << mainRowVec << std::endl;
    // Finally main kernel calculates the GEM for colPointerMain from mainRowVec
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
      #ifdef HAVE_MPI
    for( int i = 0; i < numOfProcesses; i++ )
    {
      if( processID == i )
      {
        showMatrix<<< 1, 1 >>>( this->A );
        cudaDeviceSynchronize();
        TNL_CHECK_CUDA_DEVICE;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    for( int i = 0; i < numOfProcesses; i++ )
    {
      if( i == processID )
        std::cout << device_vector;
      MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << std::endl;
#else
    showMatrix<<< 1, 1 >>>( this->A );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
    std::cout << this->b << std::endl;
    printf("\n");
#endif
    }
    // increment colPointerMain for next while passage
    colPointerMain++;
  }
  
  // delete all used variables
  cudaFree(pivot);
  cudaFree( devMat );
  TNL_CHECK_CUDA_DEVICE;
  
  // Calculate real result 
  // (With MPI needs to send info into process 0 as main process with real result, rest processes has result as vector of zeros)
  calculResultVector( this->A, device_vector, x, processID, numOfProcesses );
  //if( processID == 0 )
  //  std::cout << x << std::endl;
  return true;
}

#endif // HAVE_CUDA
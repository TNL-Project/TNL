#include <iostream>
#include <cstdlib>
#include <string> // input from cmd
#include <ctime> // time of computation mesurement
#include <fstream> // saving and loading vector.txt

#include "TNL/tnl-dev/src/TNL/Devices/Host.h"
#include "TNL/tnl-dev/src/TNL/Devices/Cuda.h"
#include "TNL/tnl-dev/src/TNL/Config/ConfigDescription.h"
#include "TNL/tnl-dev/src/TNL/Config/ParameterContainer.h"

#include "gem.h"

#ifdef HAVE_MPI
#include <mpi.h>
#include <stdio.h>
#endif

using namespace TNL;

void setupConfig( TNL::Config::ConfigDescription & config )
{
   config.addDelimiter( "Gaussian Elimination Method setting:" );
   config.addRequiredEntry< String >( "input-matrix", "Input matrix file name (mtx)." );
   config.addEntry< String >( "input-vector", "Input vector file name (txt). None for result as vector of ones.", "none" );
      
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "all" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< String >( "device", "Device to compute on.", "both");
   config.addEntryEnum( "CPU" );
   config.addEntryEnum( "GPU" );
   config.addEntryEnum( "both" );
   config.addEntry< String >( "pivoting", "You can use pivoting in GEM computation.", "yes" );
   config.addEntryEnum( "yes" );
   config.addEntryEnum( "no" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 0 );
}

int main( int argc, char* argv[] )
{     
#ifdef HAVE_MPI
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int processID = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &processID);
  int numOfProcesses = 0;
  MPI_Comm_size( MPI_COMM_WORLD, &numOfProcesses );
  int numOfDevices = 0;
  cudaGetDeviceCount(&numOfDevices);
  if( numOfProcesses > numOfDevices && processID == 0 )
    printf("Warning: There is too many processes for computation,"
            " some processes will compute on same device!. (Processes %d, Devices %d)\n", numOfProcesses, numOfDevices);
  cudaSetDevice(processID%numOfDevices);
#endif
  
  
  Config::ParameterContainer parameters;
  Config::ConfigDescription conf_desc;

  setupConfig( conf_desc );
  
  if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }
  
  const String & matrixName = parameters.getParameter< String >( "input-matrix" );
  const String & vectorName = parameters.getParameter< String >( "input-vector" );
  const String & device = parameters.getParameter< String >( "device" );
  const String & precision = parameters.getParameter< String >( "precision" );
  const String & pivoting = parameters.getParameter< String >( "pivoting" );
  int loops = parameters.getParameter< int >( "loops" );
  int verbose = parameters.getParameter< int >( "verbose" );
  
#ifdef HAVE_MPI
  if( processID == 0 )
#endif
    printf("%20s %15s %15s %20s %15s %15s %10s %15s %15s\n", "matrix", "#rows", "#non-zeros", "vector", "device", "precision", "loops", "time", "error");

 
  if( ( precision == "all" || precision == "float" ) )
  {
    if( ( device == "CPU" || device == "both" ) )
      Vector< float, TNL::Devices::Host, int > result = 
        runGEM< float, int, TNL::Devices::Host >( matrixName, vectorName, loops, verbose, (String)"CPU", pivoting );

    if( ( device == "GPU" || device == "both" ) )
      auto result =
        runGEM< float, int, TNL::Devices::Cuda >( matrixName, vectorName, loops, verbose, (String)"GPU", pivoting );
  }
  
  if( ( precision == "all" || precision == "double" ) )
  {
    if( ( device == "CPU" || device == "both" ) )
      Vector< double, TNL::Devices::Host, int > result = 
        runGEM< double, int, TNL::Devices::Host >( matrixName, vectorName, loops, verbose, (String)"CPU", pivoting );
    if( ( device == "GPU" || device == "both" ) )
      Vector< double, TNL::Devices::Cuda, int > result = 
        runGEM< double, int, TNL::Devices::Cuda >( matrixName, vectorName, loops, verbose, (String)"GPU", pivoting );
  }
  
#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS; 
}
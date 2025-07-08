#include <iostream>
#include <cstdlib>
#include <string>   // input from cmd
#include <ctime>    // time of computation mesurement
#include <fstream>  // saving and loading vector.txt

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Config/parseCommandLine.h>

#include "gem.h"

#ifdef HAVE_MPI
   #include <mpi.h>
   #include <stdio.h>
#endif

void
setupConfig( TNL::Config::ConfigDescription& config )
{
   config.addDelimiter( "Gaussian Elimination Method setting:" );
   config.addRequiredEntry< TNL::String >( "input-matrix", "Input matrix file name (mtx)." );
   config.addEntry< TNL::String >( "input-vector", "Input vector file name (txt). None for result as vector of ones.", "none" );

   config.addEntry< TNL::String >( "precision", "Precision of the arithmetics.", "all" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< TNL::String >( "device", "Device to compute on.", "all" );
   config.addEntryEnum( "host" );
   config.addEntryEnum( "cuda" );
   config.addEntryEnum( "hip" );
   config.addEntryEnum( "all" );
   config.addEntry< TNL::String >( "pivoting", "You can use pivoting in GEM computation.", "yes" );
   config.addEntryEnum( "yes" );
   config.addEntryEnum( "no" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 0 );
}

template< typename Real >
void
resolveDevice( TNL::Config::ParameterContainer& parameters )
{
   TNL::String device = parameters.getParameter< TNL::String >( "device" );

   if( device == "host" || device == "all" ) {
      benchmarkGEM< Real, int, TNL::Devices::Host >( parameters );
   }
   if( device == "cuda" || device == "all" ) {
      benchmarkGEM< Real, int, TNL::Devices::Cuda >( parameters );
   }
   if( device == "hip" || device == "all" ) {
      benchmarkGEM< Real, int, TNL::Devices::Hip >( parameters );
   }
}

void
resolvePrecision( TNL::Config::ParameterContainer& parameters )
{
   TNL::String precision = parameters.getParameter< TNL::String >( "precision" );
   if( precision == "float" || precision == "all" ) {
      resolveDevice< float >( parameters );
      return;
   }
   if( precision == "double" || precision == "all" ) {
      resolveDevice< double >( parameters );
      return;
   }
}

int
main( int argc, char* argv[] )
{
#ifdef HAVE_MPI
   // Initialize the MPI environment
   MPI_Init( NULL, NULL );

   // Get the number of processes
   int processID = -1;
   MPI_Comm_rank( MPI_COMM_WORLD, &processID );
   int numOfProcesses = 0;
   MPI_Comm_size( MPI_COMM_WORLD, &numOfProcesses );
   int numOfDevices = 0;
   cudaGetDeviceCount( &numOfDevices );
   if( numOfProcesses > numOfDevices && processID == 0 )
      printf( "Warning: There is too many processes for computation,"
              " some processes will compute on same device!. (Processes %d, Devices %d)\n",
              numOfProcesses,
              numOfDevices );
   cudaSetDevice( processID % numOfDevices );
#endif

   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   /*const TNL::String& matrixName = parameters.getParameter< TNL::String >( "input-matrix" );
   const TNL::String& vectorName = parameters.getParameter< TNL::String >( "input-vector" );
   const TNL::String& device = parameters.getParameter< TNL::String >( "device" );
   const TNL::String& precision = parameters.getParameter< TNL::String >( "precision" );
   const TNL::String& pivoting = parameters.getParameter< TNL::String >( "pivoting" );
   int loops = parameters.getParameter< int >( "loops" );
   int verbose = parameters.getParameter< int >( "verbose" );*/

   /*#ifdef HAVE_MPI
      if( processID == 0 )
   #endif
         printf( "%20s %15s %15s %10s %20s %17s %17s %17s\n",
                 "vector",
                 "device",
                 "precision",
                 "loops",
                 "matrix",
                 "#rows",
                 "time",
                 "error" );
   */

   resolvePrecision( parameters );
   if( precision == "all" || precision == "float" ) {
      if( device == "CPU" || device == "both" )
         TNL::Containers::Vector< float, TNL::Devices::Host, int > result =
            runGEM< float, int, TNL::Devices::Host >( matrixName, vectorName, loops, verbose, ( TNL::String ) "CPU", pivoting );

      if( device == "GPU" || device == "both" )
         TNL::Containers::Vector< float, TNL::Devices::Cuda, int > result =
            runGEM< float, int, TNL::Devices::Cuda >( matrixName, vectorName, loops, verbose, ( TNL::String ) "GPU", pivoting );
   }

   if( precision == "all" || precision == "double" ) {
      if( device == "CPU" || device == "both" )
         TNL::Containers::Vector< double, TNL::Devices::Host, int > result = runGEM< double, int, TNL::Devices::Host >(
            matrixName, vectorName, loops, verbose, ( TNL::String ) "CPU", pivoting );
      if( device == "GPU" || device == "both" )
         TNL::Containers::Vector< double, TNL::Devices::Cuda, int > result = runGEM< double, int, TNL::Devices::Cuda >(
            matrixName, vectorName, loops, verbose, ( TNL::String ) "GPU", pivoting );
   }

#ifdef HAVE_MPI
   MPI_Finalize();
#endif

   return EXIT_SUCCESS;
}

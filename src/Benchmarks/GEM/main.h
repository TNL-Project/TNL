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

using namespace TNL;

/*void printHelp(); 

void setInput( int argc, char* argv[], string& matrixName, string& vectorName, int& loops  );

template <typename real>
void calculHostVecOne(Matrix< real, Devices::Host, int>& matrix, Vector< real, Devices::Host, int >& vector, string vectorName );

template <typename real>
void readVector( Vector< real, Devices::Host, int >& host_vector, string vectorName );*/

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
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
}

int main( int argc, char* argv[] )
{     
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
  
  
  
  if( ( precision == "all" || precision == "float" ) )
  {
    if( ( device == "CPU" || device == "both" ) )
      Vector< float, TNL::Devices::Host, int > result = 
        runGEM< float, int, TNL::Devices::Host >( matrixName, vectorName, loops, verbose, (String)"CPU", pivoting );
    if( ( device == "GPU" || device == "both" ) )
      Vector< float, TNL::Devices::Cuda, int > result = 
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
  
  return EXIT_SUCCESS;
}

/*void setInput( int argc, char* argv[], string& matrixName, string& vectorName, int& loops )
{  
  if( argc == 1 )
  {
    string pom("comsol.mtx");
    matrixName = pom; 
    string pom1("comsol.txt");
    vectorName = pom1;
    loops = 1;
  }
  
  //for(int i = 0; i < argc; i++)
  //{
  //  cout<< argc << " " << argv[i] <<endl;
  //}  
  
  if( argc != 7 )
  {
    printf( "You need to put all arguments in function call\n");
    printHelp();
  } else {
    
    //if( argv[1] != (char*)"--input-matrix" || argv[3] != (char*)"--input-vector" || argv[5] != (char*)"--loops" )
    // {
    // cout << "You need to set all parameters in the same order like help." << endl;
    // printHelp();
    // }
    string pom(argv[2]);
    matrixName = pom;
    string pom1(argv[4]);
    vectorName = pom1;
    loops = stoi(argv[6]);
  }
    
  
  cout << "Setting values: \nMatrix ... " << matrixName << endl
          << "Vector ... " << vectorName << endl
          << "Loops ... " << loops << endl;
}*/

/*void printHelp()
{
  cout << "Parameter:" << setw(30) << "description:" << endl;
  cout << "--input-matrix" << setw(60) << ".mtx file placed in test-matrices foulder." << endl;
  cout << "--input-vector" << setw(60) << ".txt file placed in test-matrices foulder." << endl;
  cout << "--loops" << setw(92) << "int number of loops of calculation for computation time mesurement." << endl;
  //cout << "--real" << setw(60) << "float/double default float." << endl;
}*/
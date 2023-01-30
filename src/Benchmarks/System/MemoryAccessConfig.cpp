/***************************************************************************
                          MemoryAccessConfig.cpp  -  description
                             -------------------
    begin                : 2015/01/20
    copyright            : (C) 2015 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <cstring>
#include <cstdlib>
#include <iostream>
#include "MemoryAccessConfig.h"

using std::cout;
using std::cerr;
using std::endl;

MemoryAccessConfig config;

MemoryAccessConfig::MemoryAccessConfig()
: outputFile( "memory-access-test.txt" ),
  testType( "sequential" ),
  sequentialStride( 1 ),
  elementSize( 1 ),
  initialArraySize( 1024 ),
  maxArraySize( 1<<30 ),
  elementsPerTest( 1<<28 ),
  tlbTestBlockSize( 0 ),
  numThreads( 1 ),
  readTest( false ),
  writeTest( false ),
  accessCentralData( false ),
  interleaving( false ),
  verbose( true )
{
}

bool MemoryAccessConfig::parseCommandLineArguments( int argc, char* argv[] )
{
   int i( 0 );
   for( int i = 1; i < argc; i++ )
   {
      if( strcmp( argv[ i ], "--output-file" ) == 0 ||
          strcmp( argv[ i ], "-o" ) == 0 )
      {
         this->outputFile = argv[ ++i ];
         continue;
      }
      if( strcmp( argv[ i ], "--test-type" ) == 0  )
      {
         this->testType = argv[ ++i ];
         continue;
      }
      if( strcmp( argv[ i ], "--sequential-stride" ) == 0  )
      {
         this->sequentialStride = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--element-size" ) == 0  )
      {
         this->elementSize = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--initial-array-size" ) == 0 )
      {
         this->initialArraySize = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--max-array-size" ) == 0 )
      {
         this->maxArraySize = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--elements-per-test" ) == 0 )
      {
         this->elementsPerTest = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--tlb-test-block-size" ) == 0 )
      {
         this->tlbTestBlockSize = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--threads" ) == 0 )
      {
         this->numThreads = atoi( argv[ ++i ] );
         continue;
      }
      if( strcmp( argv[ i ], "--read-test" ) == 0 )
      {
         this->readTest = true;
         continue;
      }
      if( strcmp( argv[ i ], "--write-test" ) == 0 )
      {
         this->writeTest = true;
         continue;
      }
      if( strcmp( argv[ i ], "--access-central-data" ) == 0 )
      {
         this->accessCentralData = true;
         continue;
      }
      if( strcmp( argv[ i ], "--interleaving" ) == 0 )
      {
         this->interleaving = true;
         continue;
      }
      if( strcmp( argv[ i ], "--no-interleaving" ) == 0 )
      {
         this->interleaving = false;
         continue;
      }

      if( strcmp( argv[ i ], "--no-verbose" ) == 0 )
      {
         this->verbose = false;
         continue;
      }

      if( strcmp( argv[ i ], "--help" ) == 0 )
      {
         printHelp( argv[ 0 ] );
         return false;
      }
      std::cerr << "Unknown command line argument " << argv[ i ] << ". Use --help for more information." <<  std::endl;
      return false;
   }
   return true;
}

void MemoryAccessConfig::printHelp( const char* programName )
{
   std::cout << "Use of " << programName << ":" << std::endl
        << std::endl
        << "  --output-file or -o       Name of the output file with the test result. Can be processed with Gnuplot." << std::endl
        << "  --test-type               Test type can be random or sequential." << std::endl
        << "  --sequential-stride       Stride of the sequential test." << std::endl
        << "  --element-size            Array element size can be power of two i.e. 1, 2, 4, 8, 16, 32, 64, 128 or 256." << std::endl
        << "  --initial-array-size      Size in bytes of the test array in the first loop of the test." << std::endl
        << "  --max-array-size          Maximum size in bytes of the test array." << std::endl
        << "  --elements-per-test       If the test array has n elements then the test loop will be repeated ceil((elements-per-test)/n) times." << std::endl
        << "  --tlb-test-block-size     The random access test will proceed by given number of pages. Zero means completely random. -1 means the worst TLB scenario" << std::endl
        << "  --threads                 Number of threads. It is 1 by default." << std::endl
        << "  --read-test               Read data (first if --access-central-data is not used) from each element." << std::endl
        << "  --write-test              Write data (first if --access-central-data is not used) from each element." << std::endl
        << "  --access-central-data     Read or write the data in the middle of the array element (only with --read-test or --write-test)." << std::endl
        << "  --interleaving            The elements in the sequential multithreaded test will be interleaved." << std::endl
        << "  --no-verbose              Turns off the verbose mode." << std::endl;
}


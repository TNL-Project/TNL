/***************************************************************************
                          memory-access-test.cpp  -  description
                             -------------------
    begin                : 2015/02/04
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
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <MemoryAccessConfig.h>
#include <TestArray.h>
#include <CacheLineSize.h>
#include <CPUCyclesCounter.h>
#include <PapiCounter.h>
#include <TimerRT.h>

using std::fstream;
using std::ios;
using std::flush;
using std::setw;

void writeTableHeader( std::ostream& str )
{
   str << std::right;
   str << "#";
   str << std::setw( 140 ) << std::setfill( '-') << "-" << std::endl;
   str << "#";
   str << std::setfill( ' ' );
   str << std::setw( 40 ) << "Array size             "
       << std::setw( 20 ) << "Time"
       << std::setw( 40 ) << "CPU tics             "
       << std::setw( 20 ) << "Bandwidth"
       << std::setw( 20 ) << "L1 Cache eff."<< std::endl;
   str << "#";
   str << std::setw( 20 ) << "(KBytes)"
       << std::setw( 20 ) << "(log2 of KBytes)"
       << std::setw( 20 ) << "(microsecs./el.)"
       << std::setw( 20 ) << "(tics/el.)"
       << std::setw( 20 ) << "(log10 of tics/el.)"
       << std::setw( 20 ) << "(GB/sec)"
       << std::setw( 20 ) << "%"
       << std::endl;
   str << "#";
   str << std::setw( 140 ) << std::setfill( '-') << "-" << std::endl;
   str << std::setfill( ' ' );
}

void writeTableLine( std::ostream& str,
                     const unsigned long long int size,
                     const unsigned long long int testedElements,
                     const double& time,
                     const unsigned long long int cpuCycles,
                     const double& l1CacheEfficiency,
                     bool readWriteTest )
{
   const double sizeInKBytes = ( double ) size / ( double ) 1024.0;
   double bandwidth = sizeof( long int ) * testedElements / time / 1.0e9;
   if( readWriteTest ) bandwidth *= 2.0;
   str << std::setw( 20 ) << sizeInKBytes
       << std::setw( 20 ) << log2( sizeInKBytes )
       << std::setw( 20 ) << 1000000 * time / ( double ) testedElements
       << std::setw( 20 ) << cpuCycles / ( double ) testedElements
       << std::setw( 20 ) << log10( cpuCycles / ( double ) testedElements )
       << std::setw( 20 ) << bandwidth
       << std::setw( 20 ) << 100.0 * l1CacheEfficiency << std::endl;
}

void writeTableLastLine( std::ostream& str )
{
   str << "#";
   str << std::setw( 140 ) << std::setfill( '-') << "-" << std::endl;
   str << std::setfill( ' ' );
}

template< int Size >
int performTest( const MemoryAccessConfig& config )
{
   /****
    * Print the system information
    */
   std::cout << "Size of long int is " << sizeof( long int ) << " bytes on this system." << std::endl;
   const size_t cacheLineSize = cache_line_size();
   std::cout << "The cache line size is " << cacheLineSize << " bytes on this system." << std::endl;
   std::cout << "Size of array element is " << sizeof( ArrayElement< Size > )
        << " bytes i.e. " << ( double ) sizeof( ArrayElement< Size > ) / ( double ) cacheLineSize << " cache lines." << std::endl;
   std::cout << "Performing the " << config.testType << " test";
   if( config.readTest )
      std::cout << " with reading";
   if( config.writeTest )
      std::cout << " with writing";
   if( ( config.readTest || config.writeTest ) && config.accessCentralData )
      std::cout << " of the last data";
   if( strcmp( config.testType, "random" ) == 0 )
      std::cout << " TLB block size is " << config.tlbTestBlockSize;
   std::cout << " with " << config.numThreads << " threads";
   if( strcmp( config.testType, "sequential" ) == 0 && config.interleaving )
      std::cout << " with interleaving ";
   std::cout << std::endl;

   /****
    * Open the output file for the test results
    */
   std::fstream outputFile;
   if( config.outputFile )
   {
      outputFile.open( config.outputFile, ios::out );
      if( ! outputFile )
      {
         std::cerr << "I am not able to open the test output file " << config.outputFile << std::endl;
         return EXIT_FAILURE;
      }
   }

   if( config.verbose )
      writeTableHeader( std::cout );
   writeTableHeader( outputFile );

   /****
    * Loop over different array sizes
    */
   unsigned long long int size = config.initialArraySize;
   while( size < config.maxArraySize )
   {
      /****
       * Setup the array
       */
      TestArray< Size > testArray( size );
      if( strcmp( config.testType, "random" ) == 0 )
      {
         std::cout << "Setting up the random memory access test...    \r" << flush;
         if( ! testArray.setupRandomTest( config.tlbTestBlockSize, config.numThreads ) )
            return EXIT_FAILURE;
      }
      else if( strcmp( config.testType, "sequential" ) == 0 )
      {
         std::cout << "Setting up the sequential memory access test...   \r" << flush;
         testArray.setupSequentialTest( config.numThreads, config.interleaving );
      }
      else
      {
         std::cerr << "The test type " << config.testType << " is not known. Can be only random or sequential." << std::endl;
         return EXIT_FAILURE;
      }

      /****
       * Perform the test
       */
      if( config.verbose )
      {
         std::cout << "Performing the test with size " << size / 1024 << " kBytes.";
         std::cout << " ...        \r" << flush;
      }
      CPUCyclesCounter cpuCycles;
      TimerRT timer;
      PapiCounter papiCounter;
#ifdef HAVE_PAPI
      papiCounter.setNumberOfEvents( 2 );
      papiCounter.addEvent( PAPI_L1_DCA );
      papiCounter.addEvent( PAPI_L1_DCH );
#endif
      papiCounter.reset();
      papiCounter.start();

      timer.reset();
      timer.start();

      cpuCycles.reset();
      cpuCycles.start();
      unsigned long long int testedElements = testArray.performTest( config );

      timer.stop();
      cpuCycles.stop();
      papiCounter.stop();

      /****
       * Store the measured results
       */
      const double time = timer.getTime();
      double l1CacheEfficiency( -1.0 );
#ifdef HAVE_PAPI
      l1CacheEfficiency = ( double ) papiCounter.getEventValue( PAPI_L1_DCH ) / ( double ) papiCounter.getEventValue( PAPI_L1_DCA );
#endif

      if( config.verbose )
      {
         std::cout << "                                                                                                                                          \r" << flush;
         writeTableLine( std::cout, size, testedElements, time, cpuCycles.getCycles(), l1CacheEfficiency, config.readTest || config.writeTest );
      }
      writeTableLine( outputFile, size, testedElements, time, cpuCycles.getCycles(), l1CacheEfficiency, config.readTest || config.writeTest );

      size *= 2;
   }
   if( config.verbose )
      writeTableLastLine( std::cout );
   writeTableLastLine( outputFile );
   outputFile.close();
   return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
   MemoryAccessConfig config;
   if( ! config.parseCommandLineArguments( argc, argv ) )
      return EXIT_FAILURE;

   std::cout << "Memory access test" << std::endl;

   switch( config.elementSize )
   {
      case 1:
         return performTest< 1 >( config );
      case 2:
         return performTest< 2 >( config );
      case 4:
         return performTest< 4 >( config );
      case 8:
         return performTest< 8 >( config );
      case 16:
         return performTest< 16 >( config );
      case 32:
         return performTest< 32 >( config );
      case 64:
         return performTest< 64 >( config );
      case 128:
         return performTest< 128 >( config );
      case 256:
         return performTest< 256 >( config );
   }
   std::cerr << "Element size " << config.elementSize << " is not allowed. It can be only 1, 2, 4, 8, 16, 32, 64, 128, 256." << std::endl;
   return EXIT_FAILURE;
}


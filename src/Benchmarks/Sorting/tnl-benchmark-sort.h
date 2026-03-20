// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <iomanip>

#include "generators.h"
#include "Measurer.h"

#ifndef LOW_POW
   #define LOW_POW 10
#endif

#ifndef HIGH_POW
   #define HIGH_POW 25
#endif

#ifndef TRIES
   #define TRIES 20
#endif

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;

template< typename Sorter >
void
start( std::ostream& out, const std::string& delim )
{
   out << "size" << delim;
   out << "random" << delim;
   out << "shuffle" << delim;
   out << "sorted" << delim;
   out << "almost" << delim;
   out << "decreasing" << delim;
   out << "gauss" << delim;
   out << "bucket" << delim;
   out << "stagger" << delim;
   out << "zero_entropy";
   out << '\n';

   int wrongAnsCnt = 0;

   for( int pow = LOW_POW; pow <= HIGH_POW; pow++ ) {
      int size = ( 1 << pow );
      std::vector< int > vec( size );

      out << "2^" << pow << delim << std::flush;
      out << std::fixed << std::setprecision( 3 );

      out << Measurer< Sorter >::measure( generateRandom( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateShuffle( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateSorted( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateAlmostSorted( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateDecreasing( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateGaussian( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateBucket( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateStaggered( size ), TRIES, wrongAnsCnt );
      out << delim << std::flush;

      out << Measurer< Sorter >::measure( generateZero_entropy( size ), TRIES, wrongAnsCnt );
      out << '\n';
   }

   if( wrongAnsCnt > 0 )
      std::cerr << wrongAnsCnt << "tries were sorted incorrectly\n";
}

int
main( int argc, char* argv[] )
{
   if( argc == 1 ) {
#ifdef __CUDACC__
      std::cout << "Quicksort on GPU ...\n";
      start< experimental::Quicksort >( std::cout, "\t" );
      std::cout << "Bitonic sort on GPU ...\n";
      start< BitonicSort >( std::cout, "\t" );

   #if defined( __CUDACC__ )
      #ifdef HAVE_CUDA_SAMPLES
      std::cout << "Manca quicksort on GPU ...\n";
      start< MancaQuicksort >( std::cout, "\t" );
      std::cout << "Nvidia bitonic sort on GPU ...\n";
      start< NvidiaBitonicSort >( std::cout, "\t" );
      #endif
      std::cout << "Cederman quicksort on GPU ...\n";
      start< CedermanQuicksort >( std::cout, "\t" );
      std::cout << "Thrust radixsort on GPU ...\n";
      start< ThrustRadixsort >( std::cout, "\t" );
   #endif
#endif

      std::cout << "STL sort on CPU ...\n";
      start< STLSort >( std::cout, "\t" );
   }
   else {
      std::ofstream out( argv[ 1 ] );
#ifdef __CUDACC__
      std::cout << "Quicksort on GPU ...\n";
      start< experimental::Quicksort >( out, "," );
      std::cout << "Bitonic sort on GPU ...\n";
      start< BitonicSort >( out, "," );

   #if defined( __CUDACC__ )
      #ifdef HAVE_CUDA_SAMPLES
      std::cout << "Manca quicksort on GPU ...\n";
      start< MancaQuicksort >( out, "," );
      std::cout << "Nvidia bitonic sort on GPU ...\n";
      start< NvidiaBitonicSort >( out, "," );
      #endif
      std::cout << "Cederman quicksort on GPU ...\n";
      start< CedermanQuicksort >( out, "," );
      std::cout << "Thrust radixsort on GPU ...\n";
      start< ThrustRadixsort >( out, "," );
   #endif
#endif

      std::cout << "STL sort on CPU ...\n";
      start< STLSort >( out, "," );
   }
   return 0;
}

// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits.h>
#include <sys/time.h>
#include <math.h>

#include <TNL/Containers/Array.h>


template< int Size >
struct MemoryAccessBenchmarkTestElement
{
   long int& operator[]( int i ) {
      return data[ i ];
   }

   MemoryAccessBenchmarkTestElement* next;

   // long int has the same size as a pointer on both 32 and 64 bits systems.
   long int data[ Size - 1 ];
};

template<>
struct MemoryAccessBenchmarkTestElement< 1 >
{
   long int& operator[]( int i ) {
      std::cerr << "Calling of operator [] for TestArrayElement with Size = 1 does not make sense." << std::endl;
      abort();
   }

   MemoryAccessBenchmarkTestElement* next;
};

// We do not allow array element with no data.
template<>
class MemoryAccessBenchmarkTestElement< 0 >{};

template< int Size >
class MemoryAccessBenchmarkTestArray
{
   public:

      using ElementType = MemoryAccessBenchmarkTestElement< Size >;
      using ArrayType = TNL::Containers::Array< ElementType >;
      using PtrArrayType = TNL::Containers::Array< ElementType* >;
      using ArrayView = typename ArrayType::ViewType;

      MemoryAccessBenchmarkTestArray( unsigned long long int size );

      void setThreadsCount( int threads_count );

      unsigned long long int getElementsCount() const;

      void setElementsPerTest( long long int elementsPerTest );

      void setWriteTest( bool writeTest );

      void setReadTest( bool readTest );

      void setCentralDataAccess( bool centralDataAccess );

      void setInterleaving( bool interleaving );

      bool setupRandomTest( int tlbTestBlockSize = 0,
                            const int numThreads = 1 );

      void setupSequentialTest( const int numThreads = 1,
                                bool interleaving = true );

      void performTest();

      unsigned long long int getTestedElementsCount();

      unsigned long long int getTestedElementsCountPerThread();
   protected:

      bool setupRandomTLBWorstTest();

      bool setupRandomTestBlock( const unsigned long long int blockSize,
                                 PtrArrayType& blockLink,
                                 const int numThreads = 1 );

      template< bool readTest,
                bool writeTest,
                bool accessCentralData >
      void testLoop();

      ArrayType allocation;
      ArrayView array;

      unsigned long long int numberOfElements;

      bool readTest = true, writeTest = false, accessCentralData = false, interleaving = false;

      int num_threads = 1;

      unsigned long long int elementsPerTest, testedElementsCount;

      unsigned long long int sum = 0;
};

#include "MemoryAccessBenchmarkTestArray.hpp"

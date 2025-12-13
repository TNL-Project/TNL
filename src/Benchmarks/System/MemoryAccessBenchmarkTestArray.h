// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <sys/time.h>

#include <TNL/Containers/Array.h>

template< int Size >
struct MemoryAccessBenchmarkTestElement
{
   std::uintptr_t&
   operator[]( int i )
   {
      return data[ i ];
   }

   MemoryAccessBenchmarkTestElement* next;

   // std::uintptr_t has the same size as a pointer on both 32 and 64 bits systems.
   std::uintptr_t data[ Size - 1 ];
};

template<>
struct MemoryAccessBenchmarkTestElement< 1 >
{
   std::uintptr_t&
   operator[]( int i )
   {
      throw std::logic_error( "Calling of operator [] for TestArrayElement with Size = 1 does not make sense." );
   }

   MemoryAccessBenchmarkTestElement* next;
};

// We do not allow array element with no data.
template<>
class MemoryAccessBenchmarkTestElement< 0 >
{};

template< int Size >
class MemoryAccessBenchmarkTestArray
{
public:
   using ElementType = MemoryAccessBenchmarkTestElement< Size >;
   using ArrayType = TNL::Containers::Array< ElementType >;
   using PtrArrayType = TNL::Containers::Array< ElementType* >;
   using ArrayView = typename ArrayType::ViewType;

   MemoryAccessBenchmarkTestArray( std::uintptr_t size );

   void
   setThreadsCount( int threads_count );

   [[nodiscard]] std::uintptr_t
   getElementsCount() const;

   void
   setElementsPerTest( std::uintptr_t elementsPerTest );

   void
   setWriteTest( bool writeTest );

   void
   setReadTest( bool readTest );

   void
   setCentralDataAccess( bool accessCentralData );

   void
   setInterleaving( bool interleaving );

   bool
   setupRandomTest( int tlbTestBlockSize = 0, int numThreads = 1 );

   void
   setupSequentialTest( int numThreads = 1, bool interleaving = true );

   void
   performTest();

   std::uintptr_t
   getTestedElementsCount();

   std::uintptr_t
   getTestedElementsCountPerThread();

protected:
   bool
   setupRandomTLBWorstTest();

   bool
   setupRandomTestBlock( std::uintptr_t blockSize, PtrArrayType& blockLink, int numThreads = 1 );

   template< bool readTest, bool writeTest, bool accessCentralData >
   void
   testLoop();

   ArrayType allocation;
   ArrayView array;

   std::uintptr_t numberOfElements;

   bool readTest = true;
   bool writeTest = false;
   bool accessCentralData = false;
   bool interleaving = false;

   int num_threads = 1;

   std::uintptr_t elementsPerTest;
   std::uintptr_t testedElementsCount;

   std::uintptr_t sum = 0;
};

#include "MemoryAccessBenchmarkTestArray.hpp"

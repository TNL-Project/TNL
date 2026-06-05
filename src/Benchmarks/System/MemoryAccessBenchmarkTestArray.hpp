// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <TNL/DiscreteMath.h>

#ifdef HAVE_OPENMP
   #include <omp.h>
#endif

#include "MemoryAccessBenchmarkTestArray.h"

template< int Size >
MemoryAccessBenchmarkTestArray< Size >::MemoryAccessBenchmarkTestArray( std::uintptr_t size )
{
   this->numberOfElements = TNL::roundUpDivision( size, sizeof( ElementType ) );
   this->allocation.setSize( this->numberOfElements + TNL::roundUpDivision( 4096UL, sizeof( ElementType ) ) );
   this->elementsPerTest = this->numberOfElements;

   // Align the array to the memory page boundary
   void* ptr = this->allocation.getData();
   std::size_t space = this->allocation.getSize() * sizeof( ElementType );
   void* aligned_ptr = std::align( 4096, sizeof( ElementType ), ptr, space );
   this->array.bind( static_cast< ElementType* >( aligned_ptr ), this->numberOfElements );
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setThreadsCount( int threads_count )
{
   this->num_threads = threads_count;
}

template< int Size >
std::uintptr_t
MemoryAccessBenchmarkTestArray< Size >::getElementsCount() const
{
   return this->numberOfElements;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setElementsPerTest( std::uintptr_t elementsPerTest )
{
   this->elementsPerTest = elementsPerTest;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setWriteTest( bool writeTest )
{
   this->writeTest = writeTest;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setReadTest( bool readTest )
{
   this->readTest = readTest;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setCentralDataAccess( bool accessCentralData )
{
   this->accessCentralData = accessCentralData;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setupRandomTLBWorstTest()
{
   if( 4096 % sizeof( ElementType ) != 0U ) {
      throw std::runtime_error( "Element size does not divide the page size." );
   }
   const std::uintptr_t elementsPerPage = 4096 / sizeof( ElementType );
   const std::uintptr_t numberOfPages = ceil( this->numberOfElements * sizeof( ElementType ) / 4096.0 );

   int* elementsOnPageLeft = new int[ numberOfPages ];
   char* usedElements = new char[ this->numberOfElements ];
   memset( usedElements, 0, this->numberOfElements * sizeof( char ) );
   for( std::uintptr_t i = 0; i < numberOfPages; i++ )
      elementsOnPageLeft[ i ] = elementsPerPage;
   if( this->numberOfElements % elementsPerPage != 0 )
      elementsOnPageLeft[ numberOfPages - 1 ] = this->numberOfElements % elementsPerPage;

   std::uintptr_t previousElement;
   std::uintptr_t newElement;
   std::uintptr_t pageIndex;
   usedElements[ 0 ] = 1;
   previousElement = 0;
   elementsOnPageLeft[ 0 ]--;
   pageIndex = 0;

   for( std::uintptr_t i = 1; i < this->numberOfElements; i++ ) {
      pageIndex = ( pageIndex + 1 ) % numberOfPages;
      while( elementsOnPageLeft[ pageIndex ] == 0 )
         pageIndex = ( pageIndex + 1 ) % numberOfPages;
      elementsOnPageLeft[ pageIndex ]--;

      int thisPageElements = elementsPerPage;
      if( pageIndex == numberOfPages - 1 && this->numberOfElements % elementsPerPage != 0 )
         thisPageElements = this->numberOfElements % elementsPerPage;
      newElement = rand() % elementsPerPage;

      if( pageIndex * elementsPerPage + newElement >= this->numberOfElements )
         newElement = rand() % ( this->numberOfElements % elementsPerPage );
      while( usedElements[ pageIndex * elementsPerPage + newElement ] != 0 )
         newElement = ( newElement + 1 ) % thisPageElements;
      newElement = pageIndex * elementsPerPage + newElement;

      this->array[ previousElement ].next = &this->array[ newElement ];
      this->array[ previousElement ][ 0 ] = 1;
      this->array[ previousElement ][ ( Size - 1 ) / 2 ] = 1;
      usedElements[ newElement ] = 1;
      previousElement = newElement;
   }
   this->array[ newElement ].next = nullptr;
   this->array[ newElement ][ 0 ] = 1;
   this->array[ newElement ][ ( Size - 1 ) / 2 ] = 1;
   delete[] elementsOnPageLeft;
   delete[] usedElements;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setupRandomTestBlock( std::uintptr_t blockSize, PtrArrayType& blockLink )
{
   const int numThreads = this->num_threads;
   TNL::Containers::Array< char > usedElements( blockSize, 0 );
   TNL::Containers::Array< std::uintptr_t > previousElement( numThreads, 0 );
   TNL::Containers::Array< std::uintptr_t > newElement( numThreads, 0 );

   if( blockLink[ 0 ] != nullptr )
      for( int tid = 0; tid < numThreads && static_cast< std::uintptr_t >( tid ) < blockSize; tid++ )
         blockLink[ tid ]->next = &this->array[ tid ];

   for( int tid = 0; tid < numThreads && static_cast< std::uintptr_t >( tid ) < blockSize; tid++ ) {
      newElement[ tid ] = previousElement[ tid ] = tid;
      usedElements[ tid ] = 1;
   }

   for( std::uintptr_t i = numThreads; i < blockSize; ) {
      for( int tid = 0; tid < numThreads && i < blockSize; tid++, i++ ) {
         newElement[ tid ] = rand() % blockSize;
         std::uintptr_t aux = newElement[ tid ];
         while( usedElements[ newElement[ tid ] ] != 0 ) {
            newElement[ tid ] = ( newElement[ tid ] + 1 ) % blockSize;
            if( aux == newElement[ tid ] ) {
               throw std::runtime_error( "Cannot setup random access test." );
            }
         }
         this->array[ previousElement[ tid ] ].next = &this->array[ newElement[ tid ] ];
         if( Size > 1 ) {
            this->array[ previousElement[ tid ] ][ 0 ] = previousElement[ tid ];
            this->array[ previousElement[ tid ] ][ ( Size - 1 ) / 2 ] = previousElement[ tid ];
         }
         usedElements[ newElement[ tid ] ] = 1;
         previousElement[ tid ] = newElement[ tid ];
      }
   }
   for( int tid = 0; tid < numThreads && static_cast< std::uintptr_t >( tid ) < blockSize; tid++ ) {
      this->array[ newElement[ tid ] ].next = nullptr;
      if( Size > 1 ) {
         this->array[ newElement[ tid ] ][ 0 ] = newElement[ tid ];
         this->array[ newElement[ tid ] ][ ( Size - 1 ) / 2 ] = newElement[ tid ];
      }
      blockLink[ tid ] = &( this->array[ newElement[ tid ] ] );
   }
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setupRandomTest( int tlbTestBlockSize )
{
   const int numThreads = this->num_threads;
   srand( time( nullptr ) );

   TNL::Containers::Array< ElementType* > blockLink( numThreads, nullptr );

   if( tlbTestBlockSize == 0 )
      this->setupRandomTestBlock( this->numberOfElements, blockLink );
   else if( tlbTestBlockSize == -1 )
      this->setupRandomTLBWorstTest();
   else {
      const std::uintptr_t elementsPerBlock = tlbTestBlockSize * 4096 / sizeof( ElementType );
      for( std::uintptr_t i = 0; i < this->numberOfElements; i += elementsPerBlock )
         this->setupRandomTestBlock( this->numberOfElements, blockLink );
   }
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setupSequentialTest( bool interleaving )
{
   const int numThreads = this->num_threads;
   if( interleaving ) {
      for( std::uintptr_t i = 0; i < this->numberOfElements; i++ ) {
         if( i + numThreads < this->numberOfElements )
            this->array[ i ].next = &this->array[ i + numThreads ];
         else
            this->array[ i ].next = nullptr;
         if( Size > 1 ) {
            this->array[ i ][ 0 ] = i;
            this->array[ i ][ ( Size - 1 ) / 2 ] = i;
         }
      }
   }
   else {
      // Each thread gets its own linked-list chain: array[tid] is the head,
      // followed by elements [numThreads, numberOfElements) split evenly.
      // Skip the setup when there are not enough elements for every thread.
      if( numThreads > 1 && static_cast< std::uintptr_t >( numThreads ) >= this->numberOfElements )
         throw std::runtime_error( "Not enough elements for the requested number of threads." );

      const int elementsPerThread = this->numberOfElements / numThreads;
      for( int tid = 0; tid < numThreads; tid++ ) {
         int firstElement = tid * elementsPerThread;
         if( tid == 0 && numThreads > 1 )
            firstElement = numThreads;
         int lastElement = ( tid + 1 ) * elementsPerThread;
         if( tid == numThreads - 1 && numThreads > 1 )
            lastElement = this->numberOfElements;
         this->array[ tid ].next = &this->array[ firstElement ];
         if( Size > 1 ) {
            this->array[ tid ][ 0 ] = tid;
            this->array[ tid ][ ( Size - 1 ) / 2 ] = tid;
         }
         for( int i = firstElement; i < lastElement; i++ ) {
            if( i == lastElement - 1 )
               this->array[ i ].next = nullptr;
            else
               this->array[ i ].next = &this->array[ i + 1 ];
            if( Size > 1 ) {
               this->array[ i ][ 0 ] = i;
               this->array[ i ][ ( Size - 1 ) / 2 ] = i;
            }
         }
      }
   }
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::performTest()
{
   if( this->readTest ) {
      if( this->accessCentralData )
         testLoop< true, false, true >();
      else
         testLoop< true, false, false >();
      return;
   }

   if( this->writeTest ) {
      if( this->accessCentralData )
         testLoop< false, true, true >();
      else
         testLoop< false, true, false >();
      return;
   }
   testLoop< false, false, false >();
}

template< int Size >
template< bool readTest, bool writeTest, bool accessCentralData >
void
MemoryAccessBenchmarkTestArray< Size >::testLoop()
{
   const std::uintptr_t elementsPerTestPerThread = this->elementsPerTest / this->num_threads + 1;
   std::uintptr_t testedElementsCount = 0;
#ifdef HAVE_OPENMP
   #pragma omp parallel num_threads( this->num_threads ), reduction( + : testedElementsCount ), if( this->num_threads > 1 )
#endif
   {
#ifdef HAVE_OPENMP
      const std::uintptr_t tid = omp_get_thread_num();
#else
      const std::uintptr_t tid = 0;
#endif
      testedElementsCount = 0;
      if( tid < this->numberOfElements )
         while( testedElementsCount < elementsPerTestPerThread ) {
            ElementType* elementPtr = &this->array[ tid ];
            int elements( 0 );
            while( elementPtr ) {
               if( Size > 1 ) {
                  if( readTest && Size > 1 ) {
                     if( accessCentralData )
                        this->sum += ( *elementPtr )[ ( Size - 1 ) / 2 ];
                     else
                        this->sum += ( *elementPtr )[ 0 ];
                  }
                  if( writeTest && Size > 1 ) {
                     if( accessCentralData )
                        ( *elementPtr )[ ( Size - 1 ) / 2 ] = 1;
                     else
                        ( *elementPtr )[ 0 ] = 1;
                  }
               }
               elementPtr = elementPtr->next;
               elements++;
            }
            testedElementsCount += elements;
         }
   }
   this->testedElementsCount = testedElementsCount;
}

template< int Size >
std::uintptr_t
MemoryAccessBenchmarkTestArray< Size >::getTestedElementsCount()
{
   return this->testedElementsCount;
}

template< int Size >
std::uintptr_t
MemoryAccessBenchmarkTestArray< Size >::getTestedElementsCountPerThread()
{
   return this->testedElementsCount / this->num_threads;
}

// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_OPENMP
   #include <omp.h>
#endif

#include "MemoryAccessBenchmarkTestArray.h"

template< int Size >
MemoryAccessBenchmarkTestArray< Size >::MemoryAccessBenchmarkTestArray( unsigned long long int size )
{
   this->numberOfElements = ceil( (double) size / (double) sizeof( ElementType ) );
   this->allocation.setSize( this->numberOfElements + 4096 / sizeof( ElementType ) + 1 );
   this->elementsPerTest = this->numberOfElements;

   // Align the array to the memory page boundary
   long int ptr = (long int) &this->allocation[ 0 ];
   long int aligned_ptr = ptr >> 12;
   aligned_ptr++;
   aligned_ptr <<= 12;
   this->array.bind( (ElementType*) aligned_ptr, this->numberOfElements );
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setThreadsCount( int threads_count )
{
   this->num_threads = threads_count;
}

template< int Size >
unsigned long long int
MemoryAccessBenchmarkTestArray< Size >::getElementsCount() const
{
   return this->numberOfElements;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setElementsPerTest( long long int elementsPerTest )
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
MemoryAccessBenchmarkTestArray< Size >::setInterleaving( bool interleaving )
{
   this->interleaving = interleaving;
}

template< int Size >
bool
MemoryAccessBenchmarkTestArray< Size >::setupRandomTLBWorstTest()
{
   if( 4096 % sizeof( ElementType ) ) {
      std::cerr << "Element size does not divide the page size" << std::endl;
      return false;
   }
   const unsigned int elementsPerPage = 4096 / sizeof( ElementType );
   const unsigned long long int numberOfPages = ceil( (double) this->numberOfElements * sizeof( ElementType ) / 4096.0 );

   int* elementsOnPageLeft = new int[ numberOfPages ];
   char* usedElements = new char[ this->numberOfElements ];
   memset( usedElements, 0, this->numberOfElements * sizeof( char ) );
   for( unsigned int i = 0; i < numberOfPages; i++ )
      elementsOnPageLeft[ i ] = elementsPerPage;
   if( this->numberOfElements % elementsPerPage != 0 )
      elementsOnPageLeft[ numberOfPages - 1 ] = this->numberOfElements % elementsPerPage;

   unsigned long long int previousElement, newElement, pageIndex;
   usedElements[ 0 ] = 1;
   previousElement = 0;
   elementsOnPageLeft[ 0 ]--;
   pageIndex = 0;

   for( unsigned long long int i = 1; i < this->numberOfElements; i++ ) {
      pageIndex = ( pageIndex + 1 ) % numberOfPages;
      while( ! elementsOnPageLeft[ pageIndex ] )
         pageIndex = ( pageIndex + 1 ) % numberOfPages;
      elementsOnPageLeft[ pageIndex ]--;

      int thisPageElements = elementsPerPage;
      if( pageIndex == numberOfPages - 1 && this->numberOfElements % elementsPerPage != 0 )
         thisPageElements = this->numberOfElements % elementsPerPage;
      newElement = rand() % elementsPerPage;

      if( pageIndex * elementsPerPage + newElement >= this->numberOfElements )
         newElement = rand() % ( this->numberOfElements % elementsPerPage );
      while( usedElements[ pageIndex * elementsPerPage + newElement ] )
         newElement = ( newElement + 1 ) % thisPageElements;
      newElement = pageIndex * elementsPerPage + newElement;

      this->array[ previousElement ].next = &this->array[ newElement ];
      this->array[ previousElement ][ 0 ] = 1;
      this->array[ previousElement ][ ( Size - 1 ) / 2 ] = 1;
      usedElements[ newElement ] = 1;
      previousElement = newElement;
   }
   this->array[ newElement ].next = NULL;
   this->array[ newElement ][ 0 ] = 1;
   this->array[ newElement ][ ( Size - 1 ) / 2 ] = 1;
   delete[] elementsOnPageLeft;
   delete[] usedElements;
   return true;
}

template< int Size >
bool
MemoryAccessBenchmarkTestArray< Size >::setupRandomTestBlock( const unsigned long long int blockSize,
                                                              PtrArrayType& blockLink,
                                                              const int numThreads )
{
   TNL::Containers::Array< char > usedElements( blockSize, 0 );
   TNL::Containers::Array< unsigned long long int > previousElement( numThreads, 0 ), newElement( numThreads, 0 );

   if( blockLink[ 0 ] != NULL )
      for( int tid = 0; tid < numThreads && tid < (int) blockSize; tid++ )
         blockLink[ tid ]->next = &this->array[ tid ];

   for( int tid = 0; tid < numThreads && tid < (int) blockSize; tid++ ) {
      newElement[ tid ] = previousElement[ tid ] = tid;
      usedElements[ tid ] = 1;
   }

   for( unsigned long long int i = numThreads; i < blockSize; ) {
      for( int tid = 0; tid < numThreads && i < blockSize; tid++, i++ ) {
         newElement[ tid ] = rand() % blockSize;
         unsigned long long int aux = newElement[ tid ];
         while( usedElements[ newElement[ tid ] ] ) {
            newElement[ tid ] = ( newElement[ tid ] + 1 ) % blockSize;
            if( aux == newElement[ tid ] ) {
               std::cerr << "Error, I cannot setup random access test." << std::endl;
               return false;
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
   for( int tid = 0; tid < numThreads && tid < (int) blockSize; tid++ ) {
      this->array[ newElement[ tid ] ].next = NULL;
      if( Size > 1 ) {
         this->array[ newElement[ tid ] ][ 0 ] = newElement[ tid ];
         this->array[ newElement[ tid ] ][ ( Size - 1 ) / 2 ] = newElement[ tid ];
      }
      blockLink[ tid ] = &( this->array[ newElement[ tid ] ] );
   }
   return true;
}

template< int Size >
bool
MemoryAccessBenchmarkTestArray< Size >::setupRandomTest( int tlbTestBlockSize, const int numThreads )
{
   srand( time( NULL ) );

   TNL::Containers::Array< ElementType* > blockLink( numThreads, 0 );

   if( tlbTestBlockSize == 0 )
      return this->setupRandomTestBlock( this->numberOfElements, blockLink, numThreads );

   if( tlbTestBlockSize == -1 )
      return this->setupRandomTLBWorstTest();

   const unsigned long long int elementsPerBlock = tlbTestBlockSize * 4096 / sizeof( ElementType );
   for( unsigned long long int i = 0; i < this->numberOfElements; i += elementsPerBlock )
      if( ! this->setupRandomTestBlock( this->numberOfElements, blockLink ) )
         return false;
   return true;
}

template< int Size >
void
MemoryAccessBenchmarkTestArray< Size >::setupSequentialTest( const int numThreads, bool interleaving )
{
   if( interleaving ) {
      for( unsigned long long int i = 0; i < this->numberOfElements; i++ ) {
         if( i + numThreads < this->numberOfElements )
            this->array[ i ].next = &this->array[ i + numThreads ];
         else
            this->array[ i ].next = NULL;
         if( Size > 1 ) {
            this->array[ i ][ 0 ] = i;
            this->array[ i ][ ( Size - 1 ) / 2 ] = i;
         }
      }
   }
   else {
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
               this->array[ i ].next = NULL;
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
   const unsigned long long int elementsPerTestPerThread = this->elementsPerTest / this->num_threads + 1;
   unsigned long long int testedElementsCount = 0;
#pragma omp parallel num_threads( this->num_threads ), reduction( + : testedElementsCount ), if( this->num_threads > 1 )
   {
#ifdef HAVE_OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      testedElementsCount = 0;
      if( (unsigned long long int) tid < this->numberOfElements )
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
unsigned long long int
MemoryAccessBenchmarkTestArray< Size >::getTestedElementsCount()
{
   return this->testedElementsCount;
}

template< int Size >
unsigned long long int
MemoryAccessBenchmarkTestArray< Size >::getTestedElementsCountPerThread()
{
   return this->testedElementsCount / this->num_threads;
}

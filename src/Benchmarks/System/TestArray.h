/***************************************************************************
                          TestArray.h  -  description
                             -------------------
    begin                : 2015/02/04
    copyright            : (C) 2015 by Tomáš Oberhuber,
                         :             Milan Lang
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TestArrayH
#define TestArrayH

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <omp.h>
#include <limits.h>
#include <sys/time.h>
#include <math.h>

#ifdef HAVE_OMP
#include <omp.h>
#endif

#include <TNL/Containers/Array.h>
#include "ArrayElement.h"
#include "MemoryAccessConfig.h"


template< int Size >
struct TestArrayElement
{
   long int& operator[]( int i ) {
      return data[ i ];
   }

   TestArrayElement* next;

   // long int has the same size as a pointer on both 32 and 64 bits systems.
   long int data[ Size - 1 ];
};

template<>
struct TestArrayElement< 1 >
{
   long int& operator[]( int i ) {
      std::cerr << "Calling of operator [] for TestArrayElement with Size = 1 does not make sense." << std::endl;
      abort();
   }

   TestArrayElement* next;
};

// We do not allow array element with no data.
template<>
class TestArrayElement< 0 >{};

template< int Size >
class TestArray
{
   public:

      using ElementType = TestArrayElement< Size >;
      using ArrayType = TNL::Containers::Array< ElementType >;
      using ArrayView = typename ArrayType::ViewType;
      TestArray( unsigned long long int size );

      bool setupRandomTest( int tlbTestBlockSize,
                            const int numThreads );

      void setupSequentialTest( const int numThreads = 1,
                                bool interleaving = true );

      unsigned long long int performTest();

   protected:

      bool setupRandomTLBWorstTest();

      bool setupRandomTestBlock( ElementType* a,
                                 const unsigned long long int blockSize,
                                 ElementType** blockLink,
                                 const int numThreads = 1 );

      template< bool readTest,
                bool writeTest,
                bool accessCentralData >
      unsigned long long int  testLoop();

      TNL::Containers::Array< ElementType > array, allocation;
      //ElementType* array;

      unsigned long long int numberOfElements;

      bool readTest = true, writeTest = false, accessCentralData = false, interleaving = false;

      int num_threads = 1;

      unsigned long long int elementsPerTest;

      unsigned long long int sum = 0;
};

template< int Size >
TestArray< Size >::TestArray( unsigned long long int size )
{
   this->numberOfElements = ceil( ( double ) size / ( double ) sizeof( ElementType ) );
   this->allocation.setSize( this->numberOfElements  + 4096 / sizeof( ElementType ) + 1 );

   // Align the array to the memory page boundary
   this->array = ArrayView( ( ElementType* ) ( ( ( ( ( long int ) &this->allocation[0] ) / 4096 ) + 1 ) * 4096 ), size );
};

template< int Size >
bool TestArray< Size >::setupRandomTLBWorstTest()
{
   if( 4096 % sizeof( ElementType ) )
   {
      std::cerr << "Element size does not divide the page size" << std::endl;
      return false;
   }
   const int elementsPerPage = 4096 / sizeof( ElementType );
   const long long int numberOfPages = ceil( ( double ) this->numberOfElements * sizeof( ElementType ) / 4096.0 );

   int* elementsOnPageLeft = new int[ numberOfPages ];
   char *usedElements = new char[ this->numberOfElements ];
   memset( usedElements, 0, this->numberOfElements * sizeof( char ) );
   for( int i = 0; i < numberOfPages; i++ )
      elementsOnPageLeft[ i ] = elementsPerPage;
   if( this->numberOfElements % elementsPerPage != 0 )
      elementsOnPageLeft[ numberOfPages - 1 ] = this->numberOfElements % elementsPerPage;

   unsigned long long int previousElement, newElement, pageIndex;
   usedElements[ 0 ] = 1;
   previousElement = 0;
   elementsOnPageLeft[ 0 ]--;
   pageIndex = 0;

   for( unsigned long long int i = 1; i < this->numberOfElements; i++)
   {
      pageIndex = ++pageIndex % numberOfPages;
      while( ! elementsOnPageLeft[ pageIndex ] )
         pageIndex = ++ pageIndex % numberOfPages;
      elementsOnPageLeft[ pageIndex ]--;

      int thisPageElements = elementsPerPage;
      if( pageIndex == numberOfPages - 1 && this->numberOfElements % elementsPerPage != 0 )
         thisPageElements = this->numberOfElements % elementsPerPage;
      newElement = rand() % elementsPerPage;

      if( pageIndex * elementsPerPage + newElement >= this->numberOfElements )
         newElement = rand() % ( this->numberOfElements % elementsPerPage );
      while( usedElements[ pageIndex * elementsPerPage + newElement ] )
         newElement = ++ newElement % thisPageElements;
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
bool TestArray< Size >::setupRandomTestBlock( ElementType* a,
                                              const unsigned long long int blockSize,
                                              ElementType** blockLink,
                                              const int numThreads )
{
   char *usedElements = new char[ blockSize ];
   memset( usedElements, 0, blockSize * sizeof( char ) );
   unsigned long long int *previousElement, *newElement;
   previousElement = new unsigned long long int[ numThreads ];
   newElement = new unsigned long long int[ numThreads ];

   if( blockLink[ 0 ] != NULL )
      for( int tid = 0; tid < numThreads && tid < blockSize; tid++ )
      blockLink[ tid ]->next = &a[ tid ];


   for( int tid = 0; tid < numThreads && tid < blockSize; tid ++ )
   {
      newElement[ tid ] = previousElement[ tid ] = tid;
      usedElements[ tid ] = 1;
   }

   for( unsigned long long int i = numThreads; i < blockSize; )
   {
      for( int tid = 0;
           tid < numThreads && i < blockSize;
           tid++, i++ )
      {
         newElement[ tid ] = rand() % blockSize;
         unsigned long long int aux = newElement[ tid ];
         while( usedElements[ newElement[ tid ] ] )
         {
            newElement[ tid ] = ( newElement[ tid ] + 1 ) % blockSize;
            if( aux == newElement[ tid ] )
            {
               std::cerr << "Error, I cannot setup random access test." << std::endl;
               return false;
            }
         }
         //cout << "New element for TID " << tid << " is " << newElement[ tid ] << std::endl;
         this->array[ previousElement[ tid ] ].next = &this->array[ newElement[ tid ] ];
         this->array[ previousElement[ tid ] ][ 0 ] = previousElement[ tid ];
         this->array[ previousElement[ tid ] ][ ( Size - 1 ) / 2 ] = previousElement[ tid ];
         usedElements[ newElement[ tid ] ] = 1;
         previousElement[ tid ] = newElement[ tid ];
      }
   }
   //cout << "Setting the last element..." << std::endl;
   for( int tid = 0; tid < numThreads && tid < blockSize; tid++ )
   {
      this->array[ newElement[ tid ] ].next = NULL;
      this->array[ newElement[ tid ] ][ 0 ] = newElement[ tid ];
      this->array[ newElement[ tid ] ][ ( Size - 1 ) / 2 ] = newElement[ tid ];
      blockLink[ tid ] = &( this->array[ newElement[ tid ] ] );
   }
   //cout << "Freeing allocated memory..." << std::endl;
   delete[] usedElements;
   delete[] newElement;
   delete[] previousElement;
   //cout << "Freeing allocated memory... done" << std::endl;
   //abort();
   return true;
}

template< int Size >
bool TestArray< Size >::setupRandomTest( int tlbTestBlockSize,
                                         const int numThreads )
{
   srand( time( NULL ) );

   ElementType** blockLink = new ElementType*[ numThreads ];
   memset( blockLink, 0, numThreads * sizeof( ElementType* ) );

   if( tlbTestBlockSize == 0 )
      return this->setupRandomTestBlock( this->array, this->numberOfElements, blockLink, numThreads );

   if( tlbTestBlockSize == -1 )
      return this->setupRandomTLBWorstTest();

   const unsigned long long int elementsPerBlock = tlbTestBlockSize * 4096 / sizeof( ElementType );
   for( unsigned long long int i = 0; i < this->numberOfElements; i += elementsPerBlock )
      if( ! this->setupRandomTestBlock( this->array, this->numberOfElements, blockLink ) )
      {
         delete[] blockLink;
         return false;
      }

   delete[] blockLink;
   return true;
}

template< int Size >
void TestArray< Size >::setupSequentialTest( const int numThreads,
                                             bool interleaving )
{
   if( interleaving )
   {
      for( unsigned long long int i = 0; i < this->numberOfElements; i++ )
      {
         if( i + numThreads < this->numberOfElements )
            this->array[ i ].next = &this->array[ i + numThreads ];
         else
            this->array[ i ].next = NULL;
         if( Size > 1 )
         {
            this->array[ i ][ 0 ] = i;
            this->array[ i ][ ( Size - 1 ) / 2 ] = i;
         }
      }
   }
   else
   {
      const int elementsPerThread = this->numberOfElements / numThreads;
      for( int tid = 0; tid < numThreads; tid++ )
      {
         int firstElement = tid * elementsPerThread;
         if( tid == 0 && numThreads > 1)
            firstElement = numThreads;
         int lastElement = ( tid + 1 ) * elementsPerThread;
         if( tid == numThreads - 1 && numThreads > 1 )
            lastElement = this->numberOfElements;
         this->array[ tid ].next = &this->array[ firstElement ];
         if( Size > 1 )
         {
            this->array[ tid ][ 0 ] = tid;
            this->array[ tid ][ ( Size - 1 ) / 2 ] = tid;
         }
         for( int i = firstElement; i < lastElement; i++ )
         {
            if( i == lastElement - 1 )
               this->array[ i ].next = NULL;
            else
               this->array[ i ].next =&this->array[ i + 1 ];
            if( Size > 1 )
            {
               this->array[ i ][ 0 ] = i;
               this->array[ i ][ ( Size - 1 ) / 2 ] = i;
            }
         }
      }
   }
}

template< int Size >
unsigned long long int TestArray< Size >::performTest()
{
   if( this->readTest ) {
      if( this->accessCentralData )
         return testLoop< true, false, true >();
      else
         return testLoop< true, false, false >();
   }

   if( this->writeTest ) {
      if( this->accessCentralData )
         return testLoop< false, true, true >();
      else
         return testLoop< false, true, false >();
   }

   return testLoop< false, false, false >();
}

template< int Size >
   template< bool readTest,
             bool writeTest,
             bool accessCentralData >
unsigned long long int TestArray< Size >::testLoop()
{
   const unsigned long long int elementsPerTestPerThread = this->elementsPerTest / this->num_threads + 1;
   unsigned long long int testedElements = 0;
#pragma omp parallel num_threads( this->num_threads ), reduction( +:testedElements ), if( this->num_threads > 1 )
   {
#ifdef HAVE_OMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      testedElements = 0;
      if( tid < this->numberOfElements )
        while( testedElements < elementsPerTestPerThread )
        {
           ElementType* elementPtr = &this->array[ tid ];
           int elements( 0 );
           while( elementPtr )
           {
               if( Size > 1 )
               {
                  if( readTest ) {
                     if( accessCentralData )
                        this->sum += ( *elementPtr )[ ( Size - 1 )/2 ];
                     else
                        this->sum += ( *elementPtr )[ 0 ];
                  }
                  if( writeTest ) {
                     if( accessCentralData )
                        ( *elementPtr )[ ( Size - 1 )/2 ] = 1;
                     else
                        ( *elementPtr )[ 0 ] = 1;
                  }
               }
               elementPtr = elementPtr->next;
               elements ++;
           }
           testedElements += elements;
        }
   }
   return testedElements;
}

#endif

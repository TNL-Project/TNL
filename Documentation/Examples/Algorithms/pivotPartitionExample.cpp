// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>

#include <TNL/Algorithms/pivotPartition.h>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

/* A value with a comparison key and a distinct payload id. The comparator
 * below orders the values by the key only, so all values with the same key
 * are equivalent, although they carry different payloads.
 */
struct KeyPayload
{
   int key;
   int id;
};

/* Equality of the values, i.e. the same key and the same payload id. This is
 * not the equivalence used by the partitioning, which is defined by the
 * comparator below.
 */
bool
operator==( const KeyPayload& left, const KeyPayload& right )
{
   return left.key == right.key && left.id == right.id;
}

/* Orders the values by the comparison key, the payload id is ignored.
 */
bool
compareKeys( const KeyPayload& left, const KeyPayload& right )
{
   return left.key < right.key;
}

using PayloadArray = Containers::Array< KeyPayload >;

void
printElements( const PayloadArray& data, int from, int to )
{
   for( int i = from; i < to; i++ )
      std::cout << " (key = " << data[ i ].key << ", id = " << data[ i ].id << ")";
   std::cout << '\n';
}

void
printBuckets( const PayloadArray& data, int lessEnd, int greaterBegin )
{
   std::cout << "   Less bucket ( " << lessEnd << " elements ):";
   printElements( data, 0, lessEnd );
   std::cout << "   Equal bucket ( " << greaterBegin - lessEnd << " elements ):";
   printElements( data, lessEnd, greaterBegin );
   std::cout << "   Greater bucket ( " << data.getSize() - greaterBegin << " elements ):";
   printElements( data, greaterBegin, data.getSize() );
}

/* Checks that the array is a permutation of the input: every original id
 * (0, ..., size-1) occurs exactly once and every element keeps the key it
 * had in the input (key = id % 3). In particular, no element was
 * reconstructed from the pivot, which would carry the pivot's id -1.
 */
bool
isPermutationOfInput( const PayloadArray& data )
{
   std::array< bool, 16 > seen{};
   for( int i = 0; i < data.getSize(); i++ ) {
      const int id = data[ i ].id;
      if( id < 0 || id >= 16 || seen[ id ] || data[ i ].key != id % 3 )
         return false;
      seen[ id ] = true;
   }
   return true;
}

int
main()
{
   /* Input of 16 elements with keys i % 3 and distinct ids, so every key
    * occurs multiple times and all elements with the same key are equivalent.
    */
   PayloadArray input( 16 );
   for( int i = 0; i < 16; i++ )
      input[ i ] = { i % 3, i };

   /* The pivot is equivalent to all elements with key = 1, but its payload
    * id does not occur in the input.
    */
   const KeyPayload pivot{ 1, -1 };

   std::cout << "Input:";
   printElements( input, 0, input.getSize() );
   std::cout << "Pivot: (key = " << pivot.key << ", id = " << pivot.id << ")\n";

   /***
    * In-place partitioning.
    */
   PayloadArray data( input );
   const auto inPlaceResult = pivotPartition( data, pivot, compareKeys );

   std::cout << "\nIn-place partitioning:\n";
   std::cout << "   lessEnd = " << inPlaceResult.lessEnd << ", greaterBegin = " << inPlaceResult.greaterBegin << '\n';
   printBuckets( data, inPlaceResult.lessEnd, inPlaceResult.greaterBegin );
   if( isPermutationOfInput( data ) )
      std::cout << "   Every element kept its original key and id, no element was reconstructed from the pivot.\n";

   /***
    * Out-of-place partitioning.
    */
   const PayloadArray inputBefore( input );
   PayloadArray output( input.getSize() );
   const auto outOfPlaceResult = pivotPartition( input, output, pivot, compareKeys );

   std::cout << "\nOut-of-place partitioning:\n";
   std::cout << "   lessEnd = " << outOfPlaceResult.lessEnd << ", greaterBegin = " << outOfPlaceResult.greaterBegin << '\n';
   printBuckets( output, outOfPlaceResult.lessEnd, outOfPlaceResult.greaterBegin );
   if( isPermutationOfInput( output ) )
      std::cout << "   Every element kept its original key and id, no element was reconstructed from the pivot.\n";
   if( input == inputBefore ) {
      std::cout << "   The input is unchanged by the out-of-place call:\n";
      std::cout << "      Input:";
      printElements( input, 0, input.getSize() );
   }
   return EXIT_SUCCESS;
}

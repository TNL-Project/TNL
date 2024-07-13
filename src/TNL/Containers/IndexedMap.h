// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <iostream>

namespace TNL::Containers {

template< typename Value, typename Index, typename Key >
class IndexedMap
{
public:
   using ValueType = Value;
   using IndexType = Index;
   using KeyType = Key;

   void
   reset();

   [[nodiscard]] IndexType
   getSize() const;

   IndexType
   insert( const ValueType& data );

   [[nodiscard]] bool
   find( const ValueType& data, IndexType& index ) const;

   template< typename ArrayType >
   void
   toArray( ArrayType& array ) const;

   [[nodiscard]] const Value&
   getElement( KeyType key ) const;

   [[nodiscard]] Value&
   getElement( KeyType key );

   void
   print( std::ostream& str ) const;

protected:
   struct DataWithIndex
   {
      // This constructor is here only because of bug in g++, we might fix it later.
      // http://stackoverflow.com/questions/22357887/comparing-two-mapiterators-why-does-it-need-the-copy-constructor-of-stdpair
      DataWithIndex() {}

      DataWithIndex( const DataWithIndex& d ) : data( d.data ), index( d.index ) {}

      explicit DataWithIndex( const Value data ) : data( data ) {}

      DataWithIndex( const Value data, const Index index ) : data( data ), index( index ) {}

      Value data;
      Index index;
   };

   using STDMapType = std::map< Key, DataWithIndex >;
   using STDMapValueType = typename STDMapType::value_type;
   using STDMapIteratorType = typename STDMapType::const_iterator;

   STDMapType map;
};

template< typename Value, typename Index, typename Key >
std::ostream&
operator<<( std::ostream& str, IndexedMap< Value, Index, Key >& set );

}  // namespace TNL::Containers

#include <TNL/Containers/IndexedMap_impl.h>

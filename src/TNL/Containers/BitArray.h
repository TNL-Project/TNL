// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <iostream>
#include <cmath>
#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Containers {

/**
 * \brief Implementation of bit array (or bitset).
 *
 * \tparam Size
 * \tparam int
 */
template< int Size, typename Base = unsigned int >
struct BitArray
{
   using BaseType = Base;

   constexpr static int
   getSize();

   constexpr static int
   getBitsPerElement();

   constexpr static int
   getDataSize();

   __cuda_callable__
   constexpr BitArray();

   __cuda_callable__
   constexpr BitArray( const BaseType& b );

   /**
    * \brief Constructor from array.
    *
    * The input array is considered as big-endian encoding of the bit array.
    * See the following example:
    *
    * \par Example
    * \include Containers/BitArrayExample_constructorWithArray.cpp
    * \par Output
    * \include BitArrayExample_constructorWithArray.out
    */
   __cuda_callable__
   constexpr BitArray( const BaseType* b, int b_size = Size );

   __cuda_callable__
   constexpr BitArray( const BitArray& array );

   __cuda_callable__
   constexpr BitArray&
   operator=( const BitArray& b );

   __cuda_callable__
   constexpr bool
   operator==( const BitArray& b ) const;

   template< typename BaseType_ >
   __cuda_callable__
   constexpr bool
   operator==( const BitArray< Size, BaseType_ >& b ) const;

   __cuda_callable__
   constexpr bool
   operator!=( const BitArray& b ) const;

   template< typename BaseType_ >
   __cuda_callable__
   constexpr bool
   operator!=( const BitArray< Size, BaseType_ >& b ) const;

   __cuda_callable__
   constexpr BaseType
   operator&( const BaseType& b ) const;

   __cuda_callable__
   constexpr BitArray&
   operator++( int );

   __cuda_callable__
   constexpr BitArray&
   operator>>=( int i );

   __cuda_callable__
   constexpr bool
   operator[]( int i ) const;

   // constexpr void operator <<= ( int i );

   __cuda_callable__
   constexpr void
   reset();

   std::ostream&
   print( std::ostream& str = std::cout ) const;

protected:

   __cuda_callable__
   constexpr void
   shiftRight();

   BaseType data[ getDataSize() ];
};

template< int Size, typename BaseType >
std::ostream&
operator<<( std::ostream& str, const BitArray< Size, BaseType >& b );

}  // namespace Containers
}  // namespace TNL

#include <TNL/Containers/BitArray.hpp>

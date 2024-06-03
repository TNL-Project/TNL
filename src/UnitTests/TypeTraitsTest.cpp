#include <TNL/TypeTraits.h>
#include <TNL/Arithmetics/Complex.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;

TEST( TypeTraitsTest, GetValueType )
{
   static_assert( std::is_same_v< GetValueType_t< float >, float > );
   static_assert( std::is_same_v< GetValueType_t< double >, double > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, double > >, double > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, float > >, float > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, Complex< double > > >, double > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, Complex< float > > >, float > );
   static_assert( std::is_same_v< GetValueType_t< Vector< double > >, double > );
   static_assert( std::is_same_v< GetValueType_t< Vector< float > >, float > );
   static_assert( std::is_same_v< GetValueType_t< Vector< Complex< double > > >, double > );
   static_assert( std::is_same_v< GetValueType_t< Vector< Complex< float > > >, float > );
}

#include "main.h"

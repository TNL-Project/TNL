#include <TNL/TypeTraits.h>
#include <TNL/Arithmetics/Complex.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;

TEST( TypeTraitsTest, GetRealType )
{
   EXPECT_TRUE( (std::is_same_v< GetRealType< float >, float >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< double >, double >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< StaticVector< 1, double > >, double >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< StaticVector< 1, float > >, float >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< StaticVector< 1, Complex< double > > >, double >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< StaticVector< 1, Complex< float > > >, float >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< Vector< double > >, double >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< Vector< float > >, float >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< Vector< Complex< double > > >, double >) );
   EXPECT_TRUE( (std::is_same_v< GetRealType< Vector< Complex< float > > >, float >) );
}

#include "main.h"

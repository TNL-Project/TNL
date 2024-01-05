#include <array>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Expressions/LinearCombination.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Expressions;
using namespace TNL::Containers::Expressions::detail;

// test fixture for typed tests
template< typename Vector >
class LinearCombinationTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using RealType = typename VectorType::RealType;
};

// types for which VectorTest is instantiated
using VectorTypes = ::testing::Types< Vector< double > >;

TYPED_TEST_SUITE( LinearCombinationTest, VectorTypes );

template< typename Value >
struct Coefficients_0
{
   static constexpr std::array< Value, 1 > array{ 0.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_1
{
   static constexpr std::array< Value, 1 > array{ 1.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_1_0
{
   static constexpr std::array< Value, 2 > array{ 1.0, 0.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_0_1
{
   static constexpr std::array< Value, 2 > array{ 0.0, 1.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_0_0
{
   static constexpr std::array< Value, 2 > array{ 0.0, 0.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_1_2
{
   static constexpr std::array< Value, 2 > array{ 1.0, 2.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_1_0_0
{
   static constexpr std::array< Value, 3 > array{ 1.0, 0.0, 0.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};
template< typename Value >
struct Coefficients_0_1_0
{
   static constexpr std::array< Value, 3 > array{ 0.0, 1.0, 0.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};
template< typename Value >
struct Coefficients_0_0_1
{
   static constexpr std::array< Value, 3 > array{ 0.0, 0.0, 1.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};
template< typename Value >
struct Coefficients_1_1_0
{
   static constexpr std::array< Value, 3 > array{ 1.0, 1.0, 0.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};
template< typename Value >
struct Coefficients_1_0_1
{
   static constexpr std::array< Value, 3 > array{ 1.0, 0.0, 1.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};
template< typename Value >
struct Coefficients_0_1_1
{
   static constexpr std::array< Value, 3 > array{ 0.0, 1.0, 1.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

template< typename Value >
struct Coefficients_1_2_3
{
   static constexpr std::array< Value, 3 > array{ 1.0, 2.0, 3.0 };

   static constexpr int
   getSize()
   {
      return array.size();
   }

   static constexpr Value
   getValue( int i )
   {
      return array[ i ];
   }
};

TYPED_TEST( LinearCombinationTest, TypeTest_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = RealType;

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_1< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_0< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_2 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_2< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 2.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_0_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_0_0< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_1_0< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_0_1< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_1_0< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_0_1< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_1_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_1_1< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_2_3 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_2_3< RealType >;
   using ResultType = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >()
                                    + ( 2.0 * std::declval< VectorType >() + 3.0 * std::declval< VectorType >() ) );

   static_assert( std::is_same< ResultType, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), result_1( size, 1.0 ), result_2( size, 1.0 );
   result_1 = LinearCombination< Coefficients_0< RealType >, VectorType >::evaluate( v1 );
   EXPECT_EQ( result_1, VectorType( size, 0.0 ) );

   std::array< VectorType, 1 > array;
   array[ 0 ] = v1;
   result_2 = LinearCombination< Coefficients_0< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, VectorType( size, 0.0 ) );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1< RealType >, VectorType >::evaluate( v1 );
   EXPECT_EQ( result_1, v1 );

   std::array< VectorType, 1 > array;
   array[ 0 ] = v1;
   result_2 = LinearCombination< Coefficients_1< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_0_1< RealType >, VectorType >::evaluate( v1, v2 );
   EXPECT_EQ( result_1, v2 );

   std::array< VectorType, 2 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   result_2 = LinearCombination< Coefficients_0_1< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1_0< RealType >, VectorType >::evaluate( v1, v2 );
   EXPECT_EQ( result_1, v1 );

   std::array< VectorType, 2 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   result_2 = LinearCombination< Coefficients_1_0< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_2 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1_2< RealType >, VectorType >::evaluate( v1, v2 );
   EXPECT_EQ( result_1, v1 + 2.0 * v2 );

   std::array< VectorType, 2 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   result_2 = LinearCombination< Coefficients_1_2< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 + 2.0 * v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_0_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1_0_0< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v1 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_1_0_0< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_0_1_0< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v2 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_0_1_0< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_0_0_1< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v3 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_0_0_1< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v3 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1_1_0< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v1 + v2 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_1_1_0< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 + v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1_0_1< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v1 + v3 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_1_0_1< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 + v3 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_1_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_0_1_1< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v2 + v3 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_0_1_1< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v2 + v3 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_2_3 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result_1, result_2;
   result_1 = LinearCombination< Coefficients_1_2_3< RealType >, VectorType >::evaluate( v1, v2, v3 );
   EXPECT_EQ( result_1, v1 + 2.0 * v2 + 3.0 * v3 );

   std::array< VectorType, 3 > array;
   array[ 0 ] = v1;
   array[ 1 ] = v2;
   array[ 2 ] = v3;
   result_2 = LinearCombination< Coefficients_1_2_3< RealType >, VectorType >::evaluate( array );
   EXPECT_EQ( result_2, v1 + 2.0 * v2 + 3.0 * v3 );
}

#include "../main.h"

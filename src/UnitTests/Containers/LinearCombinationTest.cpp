#ifdef HAVE_GTEST
#include <array>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Expressions/LinearCombination.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Expressions;

// test fixture for typed tests
template< typename Vector >
class LinearCombinationTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using RealType = typename VectorType::RealType;
};

// types for which VectorTest is instantiated
using VectorTypes = ::testing::Types<
   Vector< double >
>;

TYPED_TEST_SUITE( LinearCombinationTest, VectorTypes );

template< typename Value >
struct Coefficients_0 {

   static constexpr std::array< Value, 1 > array{ 0.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};


template< typename Value >
struct Coefficients_1 {

   static constexpr std::array< Value, 1 > array{ 1.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

template< typename Value >
struct Coefficients_1_0 {

   static constexpr std::array< Value, 2 > array{ 1.0, 0.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

template< typename Value >
struct Coefficients_0_1 {

   static constexpr std::array< Value, 2 > array{ 0.0, 1.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

template< typename Value >
struct Coefficients_0_0 {

   static constexpr std::array< Value, 2 > array{ 0.0, 0.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

template< typename Value >
struct Coefficients_1_2 {

   static constexpr std::array< Value, 2 > array{ 1.0, 2.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

template< typename Value >
struct Coefficients_1_0_0 {

   static constexpr std::array< Value, 3 > array{ 1.0, 0.0, 0.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};
template< typename Value >
struct Coefficients_0_1_0 {

   static constexpr std::array< Value, 3 > array{ 0.0, 1.0, 0.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};
template< typename Value >
struct Coefficients_0_0_1 {

   static constexpr std::array< Value, 3 > array{ 0.0, 0.0, 1.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};
template< typename Value >
struct Coefficients_1_1_0 {

   static constexpr std::array< Value, 3 > array{ 1.0, 1.0, 0.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};
template< typename Value >
struct Coefficients_1_0_1 {

   static constexpr std::array< Value, 3 > array{ 1.0, 0.0, 1.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};
template< typename Value >
struct Coefficients_0_1_1 {

   static constexpr std::array< Value, 3 > array{ 0.0, 1.0, 1.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

template< typename Value >
struct Coefficients_1_2_3 {

   static constexpr std::array< Value, 3 > array{ 1.0, 2.0, 3.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

TYPED_TEST( LinearCombinationTest, TypeTest_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = RealType;

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_1< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_0< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_2 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_2< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 2.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_0_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_0_0< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_1_0< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_0_1< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_1_0< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_0_1< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_0_1_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_0_1_1< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest_1_2_3 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1_2_3< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + ( 2.0 * std::declval< VectorType >() + 3.0 * std::declval< VectorType >() ) );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), result( size, 1.0 );

   result = LinearCombination< Coefficients_0< RealType >, VectorType >::evaluate( v1 );

   EXPECT_EQ( result, VectorType( size, 0.0 ) );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), result;

   result = LinearCombination< Coefficients_1< RealType >, VectorType >::evaluate( v1 );

   EXPECT_EQ( result, v1 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), result;

   result = LinearCombination< Coefficients_0_1< RealType >, VectorType >::evaluate( v1, v2 );

   EXPECT_EQ( result, v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), result;

   result = LinearCombination< Coefficients_1_0< RealType >, VectorType >::evaluate( v1, v2 );

   EXPECT_EQ( result, v1 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_2 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), result;

   result = LinearCombination< Coefficients_1_2< RealType >, VectorType >::evaluate( v1, v2 );

   EXPECT_EQ( result, v1 + 2.0 * v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_0_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_1_0_0< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v1 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_0_1_0< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_0_0_1< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v3 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_1_0 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_1_1_0< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v1 + v2 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_0_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_1_0_1< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v1 + v3 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_0_1_1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_0_1_1< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v2 + v3 );
}

TYPED_TEST( LinearCombinationTest, VectorTests_1_2_3 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   result = LinearCombination< Coefficients_1_2_3< RealType >, VectorType >::evaluate( v1, v2, v3 );

   EXPECT_EQ( result, v1 + 2.0 * v2 + 3.0 * v3 );
}


#endif

#include "../main.h"

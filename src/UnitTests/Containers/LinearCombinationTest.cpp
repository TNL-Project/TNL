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
struct Coefficients_2 {

   static constexpr std::array< Value, 2 > array{ 1.0, 2.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};


template< typename Value >
struct Coefficients_3 {

   static constexpr std::array< Value, 3 > array{ 1.0, 2.0, 3.0 };

   static constexpr int getSize() { return array.size(); }

   static constexpr Value getValue( int i ) { return array[ i ]; }
};

TYPED_TEST( LinearCombinationTest, TypeTest1 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_1< RealType >;
   using LinearCombinationType = LinearCombination< Coefficients, VectorType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}


TYPED_TEST( LinearCombinationTest, TypeTest2 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_2< RealType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + 2.0 * std::declval< VectorType >() );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}

TYPED_TEST( LinearCombinationTest, TypeTest3 )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;
   using Coefficients = Coefficients_3< RealType >;
   using LinearCombinationType = LinearCombination< Coefficients, VectorType >;
   using ResultType1 = typename LinearCombinationReturnType< Coefficients, VectorType, 0 >::type;
   using ResultType2 = typename LinearCombination< Coefficients, VectorType >::ResultType;
   using TrueResultType = decltype( 1.0 * std::declval< VectorType >() + ( 2.0 * std::declval< VectorType >() + 3.0 * std::declval< VectorType >() ) );

   static_assert( std::is_same< ResultType1, TrueResultType >::value, "Wrong type." );
   static_assert( std::is_same< ResultType2, TrueResultType >::value, "Wrong type." );
}


TYPED_TEST( LinearCombinationTest, VectorTests )
{
   using VectorType = typename TestFixture::VectorType;
   using RealType = typename TestFixture::RealType;

   const int size = 10;

   VectorType v1( size, 1.0 ), v2( size, 2.0 ), v3( size, 3.0 ), result;

   //LinearCombination< Coefficients< RealType > >::evaluate( result, v1, v2, v3 );

}


#endif

#include "../main.h"

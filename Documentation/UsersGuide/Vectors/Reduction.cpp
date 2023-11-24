#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void
expressions()
{
   using RealType = float;
   using VectorType = Vector< RealType, Device >;
   using ViewType = VectorView< RealType, Device >;

   /****
    * Create vectors
    */
   const int size = 11;
   VectorType a_v( size ), b_v( size ), c_v( size );
   ViewType a = a_v.getView();
   ViewType b = b_v.getView();
   ViewType c = c_v.getView();
   a.forAllElements(
      [] __cuda_callable__( int i, RealType& value )
      {
         value = i;
      } );
   b.forAllElements(
      [] __cuda_callable__( int i, RealType& value )
      {
         value = i - 5.0;
      } );
   c = -5;

   std::cout << "a == " << a << std::endl;
   std::cout << "b == " << b << std::endl;
   std::cout << "c == " << c << std::endl;
   auto [ min_a_val, min_a_pos ] = argMin( a );
   auto [ max_a_val, max_a_pos ] = argMax( a );
   auto [ min_b_val, min_b_pos ] = argMin( b );
   auto [ max_b_val, max_b_pos ] = argMax( b );
   std::cout << "min( a ) == " << min_a_val << " at " << min_a_pos << std::endl;
   std::cout << "max( a ) == " << max_a_val << " at " << max_a_pos << std::endl;
   std::cout << "min( b ) == " << min_b_val << " at " << min_b_pos << std::endl;
   std::cout << "max( b ) == " << max_b_val << " at " << max_b_pos << std::endl;
   std::cout << "min( abs( b ) ) == " << min( abs( b ) ) << std::endl;
   std::cout << "sum( b ) == " << sum( b ) << std::endl;
   std::cout << "sum( abs( b ) ) == " << sum( abs( b ) ) << std::endl;
   std::cout << "Scalar product: ( a, b ) == " << ( a, b ) << std::endl;
   std::cout << "Scalar product: ( a + 3, abs( b ) / 2 ) == " << ( a + 3, abs( b ) / 2 ) << std::endl;
   const bool cmp = all( lessEqual( abs( a + b ), abs( a ) + abs( b ) ) );
   std::cout << "all( lessEqual( abs( a + b ), abs( a ) + abs( b ) ) ) == " << ( cmp ? "true" : "false" ) << std::endl;
   auto [ equal_val, equal_pos ] = argAny( equalTo( a, 5 ) );
   if( equal_val )
      std::cout << equal_pos << "-th element of a is equal to 5" << std::endl;
   else
      std::cout << "No element of a is equal to 5" << std::endl;
}

int
main( int argc, char* argv[] )
{
   /****
    * Perform test on CPU
    */
   std::cout << "Expressions on CPU ..." << std::endl;
   expressions< Devices::Host >();

   /****
    * Perform test on GPU
    */
#ifdef __CUDACC__
   std::cout << std::endl;
   std::cout << "Expressions on GPU ..." << std::endl;
   expressions< Devices::Cuda >();
#endif
}

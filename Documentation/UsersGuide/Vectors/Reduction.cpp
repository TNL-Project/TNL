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
   VectorType a_v( size );
   VectorType b_v( size );
   VectorType c_v( size );
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

   std::cout << "a == " << a << '\n';
   std::cout << "b == " << b << '\n';
   std::cout << "c == " << c << '\n';
   auto [ min_a_val, min_a_pos ] = argMin( a );
   auto [ max_a_val, max_a_pos ] = argMax( a );
   auto [ min_b_val, min_b_pos ] = argMin( b );
   auto [ max_b_val, max_b_pos ] = argMax( b );
   std::cout << "min( a ) == " << min_a_val << " at " << min_a_pos << '\n';
   std::cout << "max( a ) == " << max_a_val << " at " << max_a_pos << '\n';
   std::cout << "min( b ) == " << min_b_val << " at " << min_b_pos << '\n';
   std::cout << "max( b ) == " << max_b_val << " at " << max_b_pos << '\n';
   std::cout << "min( abs( b ) ) == " << min( abs( b ) ) << '\n';
   std::cout << "sum( b ) == " << sum( b ) << '\n';
   std::cout << "sum( abs( b ) ) == " << sum( abs( b ) ) << '\n';
   std::cout << "Scalar product: ( a, b ) == " << ( a, b ) << '\n';
   std::cout << "Scalar product: ( a + 3, abs( b ) / 2 ) == " << ( a + 3, abs( b ) / 2 ) << '\n';
   const bool cmp = all( lessEqual( abs( a + b ), abs( a ) + abs( b ) ) );
   std::cout << "all( lessEqual( abs( a + b ), abs( a ) + abs( b ) ) ) == " << ( cmp ? "true" : "false" ) << '\n';
   auto [ equal_val, equal_pos ] = argAny( equalTo( a, 5 ) );
   if( equal_val )
      std::cout << equal_pos << "-th element of a is equal to 5\n";
   else
      std::cout << "No element of a is equal to 5\n";
}

int
main( int argc, char* argv[] )
{
   /****
    * Perform test on CPU
    */
   std::cout << "Expressions on CPU ...\n";
   expressions< Devices::Host >();

   /****
    * Perform test on GPU
    */
#ifdef __CUDACC__
   std::cout << '\n';
   std::cout << "Expressions on GPU ...\n";
   expressions< Devices::Cuda >();
#endif
}

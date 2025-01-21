#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/compress.h>

template< typename Device >
void
compressExample()
{
   using Vector = TNL::Containers::Vector< int, Device >;

   // clang-format off
   //        0   1   2   3   4   5   6   7   8   9  10
   Vector v1{ 9, -1,  3,  0, -2,  8, -3,  7, -2,  0,  4 };
   Vector v2{ 9, -1, -3,  0, -2,  8, -3,  7, -2,  0, -4 };
   // clang-format on
   auto v1_view = v1.getView();
   auto v2_view = v2.getView();
   auto compressed_v = TNL::Algorithms::compress< Vector >( 0,
                                                            v1.getSize(),
                                                            [ = ] __cuda_callable__( int i )
                                                            {
                                                               return v1_view[ i ] > 0;
                                                            } );
   std::cout << "v1 = " << v1 << std::endl;
   std::cout << "Positions of positive numbers in v1 are: " << compressed_v << std::endl;

   auto n = TNL::Algorithms::compress(
      0,
      v2.getSize(),
      [ = ] __cuda_callable__( int i )
      {
         return v2_view[ i ] > 0;
      },
      compressed_v );
   std::cout << "v2 = " << v2 << std::endl;
   std::cout << "Number of positive numbers in v2 is: " << n << std::endl;
   std::cout << "Positions of positive numbers in v2 are (only first " << n << " numbers are valid): " << compressed_v
             << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running example on the host system: " << std::endl;
   compressExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running example on the CUDA device: " << std::endl;
   compressExample< TNL::Devices::Cuda >();
#endif
}

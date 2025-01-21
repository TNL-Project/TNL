#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/compress.h>

template< typename Device >
void
compressExample()
{
   using Vector = TNL::Containers::Vector< int, Device >;

   // clang-format off
   //        0  1  2  3  4  5  6  7  8  9 10
   Vector v1{ 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0 };
   Vector v2{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
   // clang-format on
   auto compressed_v = TNL::Algorithms::compress< Vector >( v1 );
   std::cout << "v1 = " << v1 << std::endl;
   std::cout << "Positions of marks in v1 are: " << compressed_v << std::endl;

   auto n = TNL::Algorithms::compress( v2, compressed_v );
   std::cout << "v2 = " << v2 << std::endl;
   std::cout << "Number of marks in v2 is: " << n << std::endl;
   std::cout << "Positions of marks in v2 is ( only first " << n << " numbers are valid):" << compressed_v << std::endl;

   TNL::Algorithms::compressFast( v2, compressed_v );
   std::cout << "v2 = " << v2 << std::endl;
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

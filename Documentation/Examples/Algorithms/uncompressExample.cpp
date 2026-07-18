#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/uncompress.h>

template< typename Device >
void
uncompressExample()
{
   using Vector = TNL::Containers::Vector< int, Device >;

   // clang-format off
   //                 0  1  2  3  4  5  6  7  8  9 10
   // mask should be: 1  0  0  0  0  1  0  1  1  0  0
   Vector indices{ 0, 5, 7, 8 };
   // clang-format on

   // Overload 1: pass explicit mask size, receive a new mask vector.
   auto mask = TNL::Algorithms::uncompress( indices, 11 );
   std::cout << "indices = " << indices << '\n';
   std::cout << "mask    = " << mask << '\n';

   // Overload 2: pass an existing mask vector to fill.
   Vector mask2;
   TNL::Algorithms::uncompress( indices, mask2, 11 );
   std::cout << "mask2   = " << mask2 << '\n';

   // Auto-detect mask size from max(indices)+1 (= 9 here).
   auto mask3 = TNL::Algorithms::uncompress( indices );
   std::cout << "mask3 (auto size) = " << mask3 << '\n';

   // Round-trip: compress → uncompress should reproduce the original mask.
   Vector original{ 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0 };
   auto compressed = TNL::Algorithms::compress< Vector >( original );
   auto restored = TNL::Algorithms::uncompress( compressed, original.getSize() );
   std::cout << "original  = " << original << '\n';
   std::cout << "restored  = " << restored << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running example on the host system:\n";
   uncompressExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running example on the CUDA device:\n";
   uncompressExample< TNL::Devices::Cuda >();
#endif
}

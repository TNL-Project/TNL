#include <iostream>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
createMultidiagonalMatrix()
{
   const int matrixSize = 6;

   /*
    * Setup the following matrix (dots represent zeros):
    *
    * /  4 -1 .  -1  .  . \
    * | -1  4 -1  . -1  . |
    * |  . -1  4 -1  . -1 |
    * | -1  . -1  4 -1  . |
    * |  . -1  . -1  4 -1 |
    * \  .  .  1  . -1  4 /
    *
    * The diagonals offsets are { -3, -1, 0, 1, 3 }.
    */
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix(
      matrixSize,
      { -3, -1, 0, 1, 3 },
      {
         /*
          * To set the matrix elements we first extend the diagonals to their full
          * lengths even outside the matrix (dots represent zeros and zeros are
          * artificial zeros used for memory alignment):
          *
          * 0  .  0 /  4 -1 .  -1  .  . \              -> {  0,  0,  4, -1, -1 }
          * .  0  . | -1  4 -1  . -1  . | .            -> {  0, -1,  4, -1, -1 }
          * .  .  0 |  . -1  4 -1  . -1 | .  .         -> {  0, -1,  4, -1, -1 }
          *    .  . | -1  . -1  4 -1  . | 0  .  .      -> { -1, -1,  4, -1,  0 }
          *       . |  . -1  . -1  4 -1 | .  0  .  .   -> { -1, -1,  4, -1,  0 }
          *         \  .  .  1  . -1  4 / 0  .  0  . . -> { -1, -1,  4,  0,  0 }
          */
         // clang-format off
         {  0,  0,  4, -1, -1 },
         {  0, -1,  4, -1, -1 },
         {  0, -1,  4, -1, -1 },
         { -1, -1,  4, -1,  0 },
         { -1, -1,  4, -1,  0 },
         { -1, -1,  4,  0,  0 },
         // clang-format on
      } );
   std::cout << "The matrix reads as:\n" << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Create multidiagonal matrix on CPU ...\n";
   createMultidiagonalMatrix< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Creating multidiagonal matrix on GPU (" << TNL::Backend::getBackendName() << ") ...\n";
   createMultidiagonalMatrix< TNL::Devices::GPU >();
#endif
}

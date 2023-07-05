#include <TNL/Containers/NDArray.h>

using namespace TNL::Containers;

int main()
{
//! [instantiation]
using RowMajorArray = NDArray< int,  // Value
                               SizesHolder< int, 0, 0 >,     // SizesHolder
                               std::index_sequence< 0, 1 >,  // Permutation
                               TNL::Devices::Host >;         // Device
RowMajorArray a;
//! [instantiation]

//! [allocation]
a.setSizes( 3, 4 );
//! [allocation]

//! [initialization]
int value = 0;
for( int i = 0; i < 3; i++ )
   for( int j = 0; j < 4; j++ )
      a( i, j ) = value++;
//! [initialization]

//! [output]
std::cout << "a = " << a.getStorageArray() << std::endl;
//! [output]
}

#include <cstdint>
#include <TNL/Containers/BitArray.h>

int main( int argc, char* argv[] )
{
   using BaseType = std::uint8_t;
   BaseType bit_data[]{ 0b11111111, 0b00000000 };

   TNL::Containers::BitArray< 16, BaseType > bit_array{ bit_data, 2 };

   std::cout << "The bit array is as follows: " << bit_array << std::endl;
}

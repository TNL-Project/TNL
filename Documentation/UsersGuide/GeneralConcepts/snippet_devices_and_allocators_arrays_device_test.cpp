template< typename Array >
void testDevice
{
   using Device = typename Array::DeviceType;
   if( std::is_same_v< Device, TNL::Device::Host > )
      std::cout << "Device is host CPU." << std::endl;
   if( std::is_same_v< Device, TNL::Device::Cuda > )
      std::cout << "Device is CUDA GPU." << std::endl;
}

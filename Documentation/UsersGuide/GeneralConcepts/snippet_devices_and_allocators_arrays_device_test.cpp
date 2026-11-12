template< typename Array >
void testDevice
{
   using Device = typename Array::DeviceType;
   if( std::is_same_v< Device, TNL::Devices::Host > )
      std::cout << "Device is host CPU.\n";
   if( std::is_same_v< Device, TNL::Devices::GPU > )
      std::cout << "Device is GPU (" << TNL::Backend::getBackendName() << ").\n";
}

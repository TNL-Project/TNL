// NOTE: this file must be a *.h file, because Doxygen does not highlight the syntax of
//       snippets included from a *.cu file

#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Pointers/DevicePointer.h>

// Define the tag for the MeshTypeResolver configuration
struct MyConfigTag
{};

namespace TNL::Meshes::BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >
{
   static constexpr bool enabled = false;
};

template<>
struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle >
{
   static constexpr bool enabled = true;
};

}  // namespace TNL::Meshes::BuildConfigTags

// Define the main task/function of the program
template< typename HostMesh >
bool
task( const HostMesh& hostMesh )
{
   //! [Parallel iteration CUDA]
   // Copy the mesh from host to the device
   using DeviceMesh = TNL::Meshes::Mesh< typename HostMesh::Config, TNL::Devices::Cuda >;
   DeviceMesh deviceMesh = hostMesh;

   // Create a device pointer to the device mesh
   TNL::Pointers::DevicePointer< const DeviceMesh > meshDevicePointer( deviceMesh );
   const DeviceMesh* meshPointer = &meshDevicePointer.template getData< typename DeviceMesh::DeviceType >();

   // Define and execute the kernel on the device
   auto kernel = [ meshPointer ] __cuda_callable__( typename DeviceMesh::GlobalIndexType i ) mutable
   {
      typename DeviceMesh::Cell elem = meshPointer->template getEntity< DeviceMesh::getMeshDimension() >( i );
      // [Do some work with the current cell `elem`...]
      (void) elem;
   };
   deviceMesh.template forAll< DeviceMesh::getMeshDimension() >( kernel );
   //! [Parallel iteration CUDA]

   return true;
}

int
main( int argc, char* argv[] )
{
   const std::string inputFileName = "example-triangles.vtu";

   auto wrapper = []( auto& reader, auto&& mesh ) -> bool
   {
      return task( mesh );
   };
   const bool result = TNL::Meshes::resolveAndLoadMesh< MyConfigTag, TNL::Devices::Host >( wrapper, inputFileName, "auto" );
   return static_cast< int >( ! result );
}

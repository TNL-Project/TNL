#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/MeshBuilder.h>

using namespace TNL;
using namespace TNL::Meshes;

struct MyConfigTag
{};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MyConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MyConfigTag, double > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MyConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle >{ static constexpr bool enabled = true; };
/*template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Hexahedron >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polyhedron >{ static constexpr bool enabled = true; };
*/

// Meshes are enabled only for the world dimension equal to the cell dimension.
template< typename CellTopology, int WorldDimension >
struct MeshSpaceDimensionTag< MyConfigTag, CellTopology, WorldDimension >
{ static constexpr bool enabled = WorldDimension == CellTopology::dimension; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MyConfigTag, float >{ static constexpr bool enabled = true; };
template<> struct MeshRealTag< MyConfigTag, double >{ static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MyConfigTag, int >{ static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MyConfigTag, long int >{ static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MyConfigTag, short int >{ static constexpr bool enabled = true; };

}  // namespace BuildConfigTags
}  // namespace Meshes
}  // namespace TNL

template< typename MeshConfig >
bool
createDualMesh( Mesh< MeshConfig, Devices::Host >& mesh, const std::string& fileName )
{
   using MeshType = Mesh< MeshConfig, Devices::Host >;
   using CellType = typename MeshType::Cell;
   using RealType = typename MeshType::RealType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;
   using PointType = typename MeshType::PointType;
   using VectorType = TNL::Containers::Vector< RealType, TNL::Devices::Host, GlobalIndexType >;

   // primary mesh
   const auto verticesCount = mesh.template getEntitiesCount< 0 >();
   const auto cellsCount = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();

   // dual mesh
   using DualMesh = Mesh <DefaultConfig<Topologies::Polygon>, TNL::Devices::Host>;
   using NeighborCountsArray = typename MeshBuilder< DualMesh >::NeighborCountsArray;
   MeshBuilder< DualMesh > builder;
   builder.setEntitiesCount(cellsCount, verticesCount); //reverse cause of dual-mesh

//triangle center
   for(int i = 0; i < cellsCount; i++){
         const auto cell = mesh.template getEntity< MeshType::getMeshDimension() >( i );
         const PointType center = getEntityCenter(mesh, cell);
         std::cout << center << std::endl;
         builder.setPoint(i, center);
      }

   NeighborCountsArray neighborCounts(verticesCount);
   for(int j = 0; j < verticesCount; j++){
      const auto vertex = mesh.template getEntity< 0 >( j );
      auto p = vertex.template getSuperentitiesCount< MeshType::getMeshDimension() >();
      neighborCounts[j] = p;
   }

   builder.setCellCornersCounts(neighborCounts);

   for(int j = 0; j < verticesCount; j++){
      const auto vertex = mesh.template getEntity< 0 >( j );
      auto p = vertex.template getSuperentitiesCount< MeshType::getMeshDimension() >();

      for(int c = 0; c < p; c++){
         auto i = vertex.template getSuperentityIndex< MeshType::getMeshDimension() >( c );
         std::cout << "vertex " << j << " is next to cell " << i << std::endl;
         builder.getCellSeed(j).setCornerId(c, i);
      }
   }

   DualMesh dualMesh;
   bool b = builder.build(dualMesh);
   if(b == false){
      std::cout<<"mistake"<<std::endl;
   }
   else{
      std::cout<<"ok"<<std::endl;
   }
   





   return true;
}

int
main( int argc, char* argv[] )
{
   if( argc < 2 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " filename.[tnl|ng|vtk|vtu|fpma] ..." << std::endl;
      return EXIT_FAILURE;
   }

   bool result = true;

   for( int i = 1; i < argc; i++ ) {
      const std::string fileName = argv[ i ];
      auto wrapper = [&]( auto& reader, auto&& mesh ) -> bool
      {
         return createDualMesh( mesh, fileName );
      };
      result &= resolveAndLoadMesh< MyConfigTag, Devices::Host >( wrapper, fileName );
   }

   return static_cast< int >( ! result );
}

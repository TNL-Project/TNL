#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>

template<int GridDimension, int TotalOrientation >
void checkEntityDimension( int expectation) {
   auto entityDimension = TNL::Meshes::GridEntitiesOrientations<GridDimension>::template getEntityDimension< TotalOrientation >();

   EXPECT_EQ( entityDimension, expectation) << "Grid Dimension: " << GridDimension
                                   << "Total orientation: " << TotalOrientation
                                   << "Entity Dimension: " << entityDimension;
}

TEST(GridEntitiesOrientationSuite, CheckEntityDimensionTest_1D ) {
   checkEntityDimension< 1, 0 >( 0 );
   checkEntityDimension< 1, 1 >( 1 );
}

TEST(GridEntitiesOrientationSuite, CheckEntityDimensionTest_2D ) {
   checkEntityDimension< 2, 0 >( 0 );

   checkEntityDimension< 2, 1 >( 1 );
   checkEntityDimension< 2, 2 >( 1 );

   checkEntityDimension< 2, 3 >( 2 );
}

TEST(GridEntitiesOrientationSuite, CheckEntityDimensionTest_3D ) {
   checkEntityDimension< 3, 0 >( 0 );

   checkEntityDimension< 3, 1 >( 1 );
   checkEntityDimension< 3, 2 >( 1 );
   checkEntityDimension< 3, 3 >( 1 );

   checkEntityDimension< 3, 4 >( 2 );
   checkEntityDimension< 3, 5 >( 2 );
   checkEntityDimension< 3, 6 >( 2 );

   checkEntityDimension< 3, 7 >( 3 );
}

TEST(GridEntitiesOrientationSuite, CheckEntityDimensionTest_4D ) {
   checkEntityDimension< 4,  0 >( 0 );

   checkEntityDimension< 4,  1 >( 1 );
   checkEntityDimension< 4,  2 >( 1 );
   checkEntityDimension< 4,  3 >( 1 );
   checkEntityDimension< 4,  4 >( 1 );

   checkEntityDimension< 4,  5 >( 2 );
   checkEntityDimension< 4,  6 >( 2 );
   checkEntityDimension< 4,  7 >( 2 );
   checkEntityDimension< 4,  8 >( 2 );
   checkEntityDimension< 4,  9 >( 2 );
   checkEntityDimension< 4, 10 >( 2 );

   checkEntityDimension< 4, 11 >( 3 );
   checkEntityDimension< 4, 12 >( 3 );
   checkEntityDimension< 4, 13 >( 3 );
   checkEntityDimension< 4, 14 >( 3 );

   checkEntityDimension< 4, 15 >( 4 );
}

template<int GridDimension, int EntityDimension, int EntityOrientation >
void compareNormals(const TNL::Containers::StaticVector<GridDimension, int>& expectation) {
   auto normals = TNL::Meshes::GridEntitiesOrientations<GridDimension>::template getNormals<EntityDimension, EntityOrientation>();

   EXPECT_EQ(normals, expectation) << "Grid Dimension: " << GridDimension
                                   << "Entity Orientation: " << EntityOrientation
                                   << "Entity Dimension: " << EntityDimension;
}

TEST( GridEntitiesOrientationSuite, NormalsTest_1D ) {
   compareNormals<1, 0, 0>({ 1 });
   compareNormals<1, 1, 0>({ 0 });
}

TEST( GridEntitiesOrientationSuite, NormalsTest_2D ) {
   compareNormals<2, 0, 0>({ 1, 1 });

   compareNormals<2, 1, 0>({ 0, 1 });
   compareNormals<2, 1, 1>({ 1, 0 });

   compareNormals<2, 2, 0>({ 0, 0 });
}

TEST( GridEntitiesOrientationSuite, NormalsTest_3D ) {
   compareNormals<3, 0, 0>({ 1, 1, 1 });

   compareNormals<3, 1, 0>({ 0, 1, 1 });
   compareNormals<3, 1, 1>({ 1, 0, 1 });
   compareNormals<3, 1, 2>({ 1, 1, 0 });

   compareNormals<3, 2, 0>({ 0, 0, 1 });
   compareNormals<3, 2, 1>({ 0, 1, 0 });
   compareNormals<3, 2, 2>({ 1, 0, 0 });

   compareNormals<3, 3, 0>({ 0, 0, 0 });
}

TEST( GridEntitiesOrientationSuite, NormalsTest_4D ) {
   compareNormals<4, 0, 0>({ 1, 1, 1, 1 });

   compareNormals<4, 1, 0>({ 0, 1, 1, 1 });
   compareNormals<4, 1, 1>({ 1, 0, 1, 1 });
   compareNormals<4, 1, 2>({ 1, 1, 0, 1 });
   compareNormals<4, 1, 3>({ 1, 1, 1, 0 });

   compareNormals<4, 2, 0>({ 0, 0, 1, 1 });
   compareNormals<4, 2, 1>({ 0, 1, 0, 1 });
   compareNormals<4, 2, 2>({ 0, 1, 1, 0 });
   compareNormals<4, 2, 3>({ 1, 0, 0, 1 });
   compareNormals<4, 2, 4>({ 1, 0, 1, 0 });
   compareNormals<4, 2, 5>({ 1, 1, 0, 0 });

   compareNormals<4, 3, 0>({ 0, 0, 0, 1 });
   compareNormals<4, 3, 1>({ 0, 0, 1, 0 });
   compareNormals<4, 3, 2>({ 0, 1, 0, 0 });
   compareNormals<4, 3, 3>({ 1, 0, 0, 0 });

   compareNormals<4, 4, 0>({ 0, 0, 0, 0 });
}

template<int GridDimension, int EntityDimension >
void compareNormalsRuntime( int entityOrientation, const TNL::Containers::StaticVector<GridDimension, int>& expectation) {
   TNL::Meshes::GridEntitiesOrientations<GridDimension> entitiesOrientations;
   auto normals = entitiesOrientations.template getNormals<EntityDimension >( entityOrientation );

   EXPECT_EQ(normals, expectation) << "Grid Dimension: " << GridDimension
                                   << "Entity Orientation: " << entityOrientation
                                   << "Entity Dimension: " << EntityDimension;
}

TEST( GridEntitiesOrientationSuite, NormalsRuntimeTest_1D ) {
   compareNormalsRuntime< 1, 0 >( 0, { 1 });
   compareNormalsRuntime< 1, 1 >( 0, { 0 });
}

TEST( GridEntitiesOrientationSuite, NormalsRuntimeTest_2D) {
   compareNormalsRuntime< 2, 0 >( 0, { 1, 1 } );

   compareNormalsRuntime< 2, 1 >( 0, { 0, 1 } );
   compareNormalsRuntime< 2, 1 >( 1, { 1, 0 } );

   compareNormalsRuntime< 2, 2 >( 0, { 0, 0 } );
}

TEST(GridEntitiesOrientationSuite, NormalsRuntimeTest_3D ) {
   compareNormalsRuntime< 3, 0 >( 0, { 1, 1, 1 } );

   compareNormalsRuntime< 3, 1 >( 0, { 0, 1, 1 } );
   compareNormalsRuntime< 3, 1 >( 1, { 1, 0, 1 } );
   compareNormalsRuntime< 3, 1 >( 2, { 1, 1, 0 } );

   compareNormalsRuntime< 3, 2 >( 0, { 0, 0, 1 } );
   compareNormalsRuntime< 3, 2 >( 1, { 0, 1, 0 } );
   compareNormalsRuntime< 3, 2 >( 2, { 1, 0, 0 } );

   compareNormalsRuntime< 3, 3 >( 0, { 0, 0, 0 } );
}

TEST(GridEntitiesOrientationSuite, NormalsRuntimeTest_4D ) {
   compareNormalsRuntime< 4, 0 >( 0, { 1, 1, 1, 1 } );

   compareNormalsRuntime< 4, 1 >( 0, { 0, 1, 1, 1 } );
   compareNormalsRuntime< 4, 1 >( 1, { 1, 0, 1, 1 } );
   compareNormalsRuntime< 4, 1 >( 2, { 1, 1, 0, 1 } );
   compareNormalsRuntime< 4, 1 >( 3, { 1, 1, 1, 0 } );

   compareNormalsRuntime< 4, 2 >( 0, { 0, 0, 1, 1 } );
   compareNormalsRuntime< 4, 2 >( 1, { 0, 1, 0, 1 } );
   compareNormalsRuntime< 4, 2 >( 2, { 0, 1, 1, 0 } );
   compareNormalsRuntime< 4, 2 >( 3, { 1, 0, 0, 1 } );
   compareNormalsRuntime< 4, 2 >( 4, { 1, 0, 1, 0 } );
   compareNormalsRuntime< 4, 2 >( 5, { 1, 1, 0, 0 } );

   compareNormalsRuntime< 4, 3 >( 0, { 0, 0, 0, 1 } );
   compareNormalsRuntime< 4, 3 >( 1, { 0, 0, 1, 0 } );
   compareNormalsRuntime< 4, 3 >( 2, { 0, 1, 0, 0 } );
   compareNormalsRuntime< 4, 3 >( 3, { 1, 0, 0, 0 } );

   compareNormalsRuntime< 4, 4 >( 0, { 0, 0, 0, 0 } );
}

template<int GridDimension, int EntityDimension, int... Normals >
void compareOrientationIndex( int expectation) {
   using NormalsGetterType = TNL::Meshes::NormalsGetter<int, EntityDimension, GridDimension>;
   using NormalsType = typename NormalsGetterType::NormalsType;
   auto index = TNL::Meshes::GridEntitiesOrientations<GridDimension>::template getOrientationIndex< EntityDimension, Normals... >();

   EXPECT_EQ(index, expectation) << "Grid Dimension: " << GridDimension << std::endl
                                 << "Entity Dimension: " << EntityDimension << std::endl
                                 << "Normals: " << NormalsType( Normals... ) << std::endl;
}

TEST( GridEntitiesOrientationSuite, OrientationIndexesTest_1D ) {
   //                       Grid. dim. | Entity. dim.  | Normals | Index
   compareOrientationIndex< 1,           0,              1         >( 0 );
   compareOrientationIndex< 1,           1,              0         >( 0 );
}

TEST( GridEntitiesOrientationSuite, OrientationIndexesTest_2D ) {
   //                       Grid. dim. | Entity. dim.  | Normals | Index
   compareOrientationIndex< 2,           0,              1, 1      >( 0 );

   compareOrientationIndex< 2,           1,              0, 1      >( 0 );
   compareOrientationIndex< 2,           1,              1, 0      >( 1 );

   compareOrientationIndex< 2,           2,              0, 0      >( 0 );
}

TEST( GridEntitiesOrientationSuite, OrientationIndexesTest_3D ) {
   //                       Grid. dim. | Entity. dim.  | Normals   | Index
   compareOrientationIndex< 3,           0,              1, 1, 1 >( 0 );

   compareOrientationIndex< 3,           1,              0, 1, 1 >( 0 );
   compareOrientationIndex< 3,           1,              1, 0, 1 >( 1 );
   compareOrientationIndex< 3,           1,              1, 1, 0 >( 2 );

   compareOrientationIndex< 3,           2,              0, 0, 1 >( 0 );
   compareOrientationIndex< 3,           2,              0, 1, 0 >( 1 );
   compareOrientationIndex< 3,           2,              1, 0, 0 >( 2 );

   compareOrientationIndex< 3,           3,              0, 0, 0 >( 0 );
}

TEST( GridEntitiesOrientationSuite, OrientationIndexesTest_4D ) {
   //                      Grid. dim. | Entity. dim.  | Normals     | Index
   compareOrientationIndex< 4,           0,             1, 1, 1, 1 >( 0 );

   compareOrientationIndex< 4,           1,             0, 1, 1, 1 >( 0 );
   compareOrientationIndex< 4,           1,             1, 0, 1, 1 >( 1 );
   compareOrientationIndex< 4,           1,             1, 1, 0, 1 >( 2 );
   compareOrientationIndex< 4,           1,             1, 1, 1, 0 >( 3 );

   compareOrientationIndex< 4,           2,             0, 0, 1, 1 >( 0 );
   compareOrientationIndex< 4,           2,             0, 1, 0, 1 >( 1 );
   compareOrientationIndex< 4,           2,             0, 1, 1, 0 >( 2 );
   compareOrientationIndex< 4,           2,             1, 0, 0, 1 >( 3 );
   compareOrientationIndex< 4,           2,             1, 0, 1, 0 >( 4 );
   compareOrientationIndex< 4,           2,             1, 1, 0, 0 >( 5 );

   compareOrientationIndex< 4,           3,             0, 0, 0, 1 >( 0 );
   compareOrientationIndex< 4,           3,             0, 0, 1, 0 >( 1 );
   compareOrientationIndex< 4,           3,             0, 1, 0, 0 >( 2 );
   compareOrientationIndex< 4,           3,             1, 0, 0, 0 >( 3 );

   compareOrientationIndex< 4,           4,             0, 0, 0, 0 >( 0 );
}

template<int GridDimension, int... Normals >
void compareTotalOrientationIndexFromNormals( int expectation) {
   using NormalsGetterType = TNL::Meshes::NormalsGetter<int, 0, GridDimension>;
   using NormalsType = typename NormalsGetterType::NormalsType;
   auto index = TNL::Meshes::GridEntitiesOrientations<GridDimension>::template getTotalOrientationIndex< Normals... >();

   EXPECT_EQ(index, expectation) << "Grid Dimension: " << GridDimension << std::endl
                                 << "Normals: " << NormalsType( Normals... ) << std::endl;
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesFromNormalsTest_1D ) {
   //                            Grid. dim.  | Normals | Index
   compareTotalOrientationIndexFromNormals< 1,           1         >( 0 );
   compareTotalOrientationIndexFromNormals< 1,           0         >( 1 );
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesFromNormalsTest_2D ) {
   //                            Grid. dim. | Normals | Index
   compareTotalOrientationIndexFromNormals< 2,           1, 1      >( 0 );

   compareTotalOrientationIndexFromNormals< 2,           0, 1      >( 1 );
   compareTotalOrientationIndexFromNormals< 2,           1, 0      >( 2 );

   compareTotalOrientationIndexFromNormals< 2,           0, 0      >( 3 );
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesFromNormalsTest_3D ) {
   //                            Grid. dim. | Normals   | Index
   compareTotalOrientationIndexFromNormals< 3,           1, 1, 1 >( 0 );

   compareTotalOrientationIndexFromNormals< 3,           0, 1, 1 >( 1 );
   compareTotalOrientationIndexFromNormals< 3,           1, 0, 1 >( 2 );
   compareTotalOrientationIndexFromNormals< 3,           1, 1, 0 >( 3 );

   compareTotalOrientationIndexFromNormals< 3,           0, 0, 1 >( 4 );
   compareTotalOrientationIndexFromNormals< 3,           0, 1, 0 >( 5 );
   compareTotalOrientationIndexFromNormals< 3,           1, 0, 0 >( 6 );

   compareTotalOrientationIndexFromNormals< 3,           0, 0, 0 >( 7 );
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesFromNormalsTest_4D ) {
   //                            Grid. dim. | Normals     | Index
   compareTotalOrientationIndexFromNormals< 4,           1, 1, 1, 1 >(  0 );

   compareTotalOrientationIndexFromNormals< 4,           0, 1, 1, 1 >(  1 );
   compareTotalOrientationIndexFromNormals< 4,           1, 0, 1, 1 >(  2 );
   compareTotalOrientationIndexFromNormals< 4,           1, 1, 0, 1 >(  3 );
   compareTotalOrientationIndexFromNormals< 4,           1, 1, 1, 0 >(  4 );

   compareTotalOrientationIndexFromNormals< 4,           0, 0, 1, 1 >(  5 );
   compareTotalOrientationIndexFromNormals< 4,           0, 1, 0, 1 >(  6 );
   compareTotalOrientationIndexFromNormals< 4,           0, 1, 1, 0 >(  7 );
   compareTotalOrientationIndexFromNormals< 4,           1, 0, 0, 1 >(  8 );
   compareTotalOrientationIndexFromNormals< 4,           1, 0, 1, 0 >(  9 );
   compareTotalOrientationIndexFromNormals< 4,           1, 1, 0, 0 >( 10 );

   compareTotalOrientationIndexFromNormals< 4,           0, 0, 0, 1 >( 11 );
   compareTotalOrientationIndexFromNormals< 4,           0, 0, 1, 0 >( 12 );
   compareTotalOrientationIndexFromNormals< 4,           0, 1, 0, 0 >( 13 );
   compareTotalOrientationIndexFromNormals< 4,           1, 0, 0, 0 >( 14 );

   compareTotalOrientationIndexFromNormals< 4,           0, 0, 0, 0 >( 15 );
}

template<int GridDimension, int EntityDimension >
void compareTotalOrientationIndex( int orientation, int expectation) {
   using NormalsGetterType = TNL::Meshes::NormalsGetter<int, 0, GridDimension>;
   using NormalsType = typename NormalsGetterType::NormalsType;
   auto index = TNL::Meshes::GridEntitiesOrientations<GridDimension>::template getTotalOrientationIndex< EntityDimension >( orientation );

   EXPECT_EQ(index, expectation) << "Grid Dimension: " << GridDimension << std::endl
                                 << "EntityDimension: " << EntityDimension << std::endl
                                 << "Orientation idx.: " << orientation << std::endl;
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesTest_1D ) {
   //                            Grid. dim.  | Entity dim. | Indexes
   compareTotalOrientationIndex< 1,           0            >( 0, 0 );
   compareTotalOrientationIndex< 1,           1            >( 0, 1 );
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesTest_2D ) {
   //                            Grid. dim.  | Entity dim. | Indexes
   compareTotalOrientationIndex< 2,           0            >( 0, 0 );

   compareTotalOrientationIndex< 2,           1            >( 0, 1 );
   compareTotalOrientationIndex< 2,           1            >( 1, 2 );

   compareTotalOrientationIndex< 2,           2            >( 0, 3 );
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesTest_3D ) {
   //                            Grid. dim.  | Entity dim. | Indexes
   compareTotalOrientationIndex< 3,           0            >( 0, 0 );

   compareTotalOrientationIndex< 3,           1            >( 0, 1 );
   compareTotalOrientationIndex< 3,           1            >( 1, 2 );
   compareTotalOrientationIndex< 3,           1            >( 2, 3 );

   compareTotalOrientationIndex< 3,           2            >( 0, 4 );
   compareTotalOrientationIndex< 3,           2            >( 1, 5 );
   compareTotalOrientationIndex< 3,           2            >( 2, 6 );

   compareTotalOrientationIndex< 3,           3            >( 0, 7 );
}

TEST( GridEntitiesOrientationSuite, TotalOrientationIndexesTest_4D ) {
   //                            Grid. dim.  | Entity dim. | Indexes
   compareTotalOrientationIndex< 4,           0            >( 0, 0 );

   compareTotalOrientationIndex< 4,           1            >( 0, 1 );
   compareTotalOrientationIndex< 4,           1            >( 1, 2 );
   compareTotalOrientationIndex< 4,           1            >( 2, 3 );
   compareTotalOrientationIndex< 4,           1            >( 3, 4 );

   compareTotalOrientationIndex< 4,           2            >( 0,  5 );
   compareTotalOrientationIndex< 4,           2            >( 1,  6 );
   compareTotalOrientationIndex< 4,           2            >( 2,  7 );
   compareTotalOrientationIndex< 4,           2            >( 3,  8 );
   compareTotalOrientationIndex< 4,           2            >( 4,  9 );
   compareTotalOrientationIndex< 4,           2            >( 5, 10 );

   compareTotalOrientationIndex< 4,           3            >( 0, 11 );
   compareTotalOrientationIndex< 4,           3            >( 1, 12 );
   compareTotalOrientationIndex< 4,           3            >( 2, 13 );
   compareTotalOrientationIndex< 4,           3            >( 3, 14 );

   compareTotalOrientationIndex< 4,           4            >( 0, 15 );
}

template<int GridDimension >
void compareNormalsTable( int totalOrientation, const TNL::Containers::StaticVector<GridDimension, int>& expectation) {
   TNL::Meshes::GridEntitiesOrientations< GridDimension > entitiesOrientations;
   auto normals = entitiesOrientations.getNormals( totalOrientation );

   EXPECT_EQ(normals, expectation) << " Grid Dimension: " << GridDimension
                                   << " Total orientation: " << totalOrientation;
}

TEST(GridEntitiesOrientationSuite, NormalsTableTest_1D ) {
   compareNormalsTable< 1 >( 0, { 1 } );
   compareNormalsTable< 1 >( 1, { 0 } );
}

TEST(GridEntitiesOrientationSuite, NormalsTableTest_2D ) {
   compareNormalsTable< 2 >( 0, { 1, 1 } );

   compareNormalsTable< 2 >( 1, { 0, 1 } );
   compareNormalsTable< 2 >( 2, { 1, 0 } );

   compareNormalsTable< 2 >( 3, { 0, 0 } );
}

TEST(GridEntitiesOrientationSuite, NormalsTableTest_3D ) {
   compareNormalsTable< 3 >( 0, { 1, 1, 1 } );

   compareNormalsTable< 3 >( 1, { 0, 1, 1 } );
   compareNormalsTable< 3 >( 2, { 1, 0, 1 } );
   compareNormalsTable< 3 >( 3, { 1, 1, 0 } );

   compareNormalsTable< 3 >( 4, { 0, 0, 1 } );
   compareNormalsTable< 3 >( 5, { 0, 1, 0 } );
   compareNormalsTable< 3 >( 6, { 1, 0, 0 } );

   compareNormalsTable< 3 >( 7, { 0, 0, 0 } );
}

TEST(GridEntitiesOrientationSuite, NormalsTableTest_4D ) {
   compareNormalsTable< 4 >( 0, { 1, 1, 1, 1 } );

   compareNormalsTable< 4 >( 1, { 0, 1, 1, 1 } );
   compareNormalsTable< 4 >( 2, { 1, 0, 1, 1 } );
   compareNormalsTable< 4 >( 3, { 1, 1, 0, 1 } );
   compareNormalsTable< 4 >( 4, { 1, 1, 1, 0 } );

   compareNormalsTable< 4 >(  5, { 0, 0, 1, 1 } );
   compareNormalsTable< 4 >(  6, { 0, 1, 0, 1 } );
   compareNormalsTable< 4 >(  7, { 0, 1, 1, 0 } );
   compareNormalsTable< 4 >(  8, { 1, 0, 0, 1 } );
   compareNormalsTable< 4 >(  9, { 1, 0, 1, 0 } );
   compareNormalsTable< 4 >( 10, { 1, 1, 0, 0 } );

   compareNormalsTable< 4 >( 11, { 0, 0, 0, 1 } );
   compareNormalsTable< 4 >( 12, { 0, 0, 1, 0 } );
   compareNormalsTable< 4 >( 13, { 0, 1, 0, 0 } );
   compareNormalsTable< 4 >( 14, { 1, 0, 0, 0 } );

   compareNormalsTable< 4 >( 15, { 0, 0, 0, 0 } );
}

template<int GridDimension, int TotalOrientation >
void compareNormalsByTotalOrientation(const TNL::Containers::StaticVector<GridDimension, int>& expectation) {
   auto normals = TNL::Meshes::GridEntitiesOrientations<GridDimension>::template getNormals< TotalOrientation >();

   EXPECT_EQ(normals, expectation) << " Grid Dimension: " << GridDimension
                                   << " Total orientation: " << TotalOrientation;
}

TEST(GridEntitiesOrientationSuite, NormalsByTotalOrientationTest_1D) {
   compareNormalsByTotalOrientation< 1, 0 >({ 1 });
   compareNormalsByTotalOrientation< 1, 1 >({ 0 });
}

TEST(GridEntitiesOrientationSuite, NormalsByTotalOrientationTest_2D ) {
   compareNormalsByTotalOrientation< 2, 0 >({ 1, 1 });

   compareNormalsByTotalOrientation< 2, 1 >({ 0, 1 });
   compareNormalsByTotalOrientation< 2, 2 >({ 1, 0 });

   compareNormalsByTotalOrientation< 2, 3 >({ 0, 0 });
}

TEST(GridEntitiesOrientationSuite, NormalsByTotalOrientationTest_3D ) {
   compareNormalsByTotalOrientation< 3, 0 >({ 1, 1, 1 });

   compareNormalsByTotalOrientation< 3, 1 >({ 0, 1, 1 });
   compareNormalsByTotalOrientation< 3, 2 >({ 1, 0, 1 });
   compareNormalsByTotalOrientation< 3, 3 >({ 1, 1, 0 });

   compareNormalsByTotalOrientation< 3, 4 >({ 0, 0, 1 });
   compareNormalsByTotalOrientation< 3, 5 >({ 0, 1, 0 });
   compareNormalsByTotalOrientation< 3, 6 >({ 1, 0, 0 });

   compareNormalsByTotalOrientation< 3, 7 >({ 0, 0, 0 });
}

TEST(GridEntitiesOrientationSuite, NormalsByTotalOrientationTest_4D) {
   compareNormalsByTotalOrientation< 4, 0 >({ 1, 1, 1, 1 });

   compareNormalsByTotalOrientation< 4,  1 >({ 0, 1, 1, 1 });
   compareNormalsByTotalOrientation< 4,  2 >({ 1, 0, 1, 1 });
   compareNormalsByTotalOrientation< 4,  3 >({ 1, 1, 0, 1 });
   compareNormalsByTotalOrientation< 4,  4 >({ 1, 1, 1, 0 });

   compareNormalsByTotalOrientation< 4,  5 >({ 0, 0, 1, 1 });
   compareNormalsByTotalOrientation< 4,  6 >({ 0, 1, 0, 1 });
   compareNormalsByTotalOrientation< 4,  7 >({ 0, 1, 1, 0 });
   compareNormalsByTotalOrientation< 4,  8 >({ 1, 0, 0, 1 });
   compareNormalsByTotalOrientation< 4,  9 >({ 1, 0, 1, 0 });
   compareNormalsByTotalOrientation< 4, 10 >({ 1, 1, 0, 0 });

   compareNormalsByTotalOrientation< 4, 11 >({ 0, 0, 0, 1 });
   compareNormalsByTotalOrientation< 4, 12 >({ 0, 0, 1, 0 });
   compareNormalsByTotalOrientation< 4, 13 >({ 0, 1, 0, 0 });
   compareNormalsByTotalOrientation< 4, 14 >({ 1, 0, 0, 0 });

   compareNormalsByTotalOrientation< 4, 15 >({ 0, 0, 0, 0 });
}

template<int GridDimension >
void testEntityDimensionFromNormals( const typename TNL::Meshes::GridEntitiesOrientations<GridDimension>::NormalsType& normals,
                                     int expectation ) {
   auto dimension = TNL::Meshes::GridEntitiesOrientations<GridDimension>::getEntityDimension( normals );

   EXPECT_EQ( dimension, expectation ) << " Grid Dimension: " << GridDimension
                                       << " Normals:" << normals;
}

TEST(GridEntitiesOrientationSuite, EntityDimensionFromNormalsTest_1D) {
   testEntityDimensionFromNormals< 1 >( { 1 }, 0 );

   testEntityDimensionFromNormals< 1 >( { 0 }, 1 );
}

TEST(GridEntitiesOrientationSuite, EntityDimensionFromNormalsTest_2D ) {
   testEntityDimensionFromNormals< 2 >( { 1, 1 }, 0 );

   testEntityDimensionFromNormals< 2 >( { 0, 1 }, 1 );
   testEntityDimensionFromNormals< 2 >( { 1, 0 }, 1 );

   testEntityDimensionFromNormals< 2 >( { 0, 0 }, 2 );
}

TEST(GridEntitiesOrientationSuite, EntityDimensionFromNormalsTest_3D ) {
   testEntityDimensionFromNormals< 3 >( { 1, 1, 1 }, 0 );

   testEntityDimensionFromNormals< 3 >( { 0, 1, 1 }, 1 );
   testEntityDimensionFromNormals< 3 >( { 1, 0, 1 }, 1 );
   testEntityDimensionFromNormals< 3 >( { 1, 1, 0 }, 1 );

   testEntityDimensionFromNormals< 3 >( { 0, 0, 1 }, 2 );
   testEntityDimensionFromNormals< 3 >( { 0, 1, 0 }, 2 );
   testEntityDimensionFromNormals< 3 >( { 1, 0, 0 }, 2 );

   testEntityDimensionFromNormals< 3 >( { 0, 0, 0 }, 3 );
}

TEST(GridEntitiesOrientationSuite, EntityDimensionFromNormalsTest_4D) {
   testEntityDimensionFromNormals< 4 >( { 1, 1, 1, 1 }, 0 );

   testEntityDimensionFromNormals< 4 >( { 0, 1, 1, 1 }, 1 );
   testEntityDimensionFromNormals< 4 >( { 1, 0, 1, 1 }, 1 );
   testEntityDimensionFromNormals< 4 >( { 1, 1, 0, 1 }, 1 );
   testEntityDimensionFromNormals< 4 >( { 1, 1, 1, 0 }, 1 );

   testEntityDimensionFromNormals< 4 >( { 0, 0, 1, 1 }, 2 );
   testEntityDimensionFromNormals< 4 >( { 0, 1, 0, 1 }, 2 );
   testEntityDimensionFromNormals< 4 >( { 0, 1, 1, 0 }, 2 );
   testEntityDimensionFromNormals< 4 >( { 1, 0, 0, 1 }, 2 );
   testEntityDimensionFromNormals< 4 >( { 1, 0, 1, 0 }, 2 );
   testEntityDimensionFromNormals< 4 >( { 1, 1, 0, 0 }, 2 );

   testEntityDimensionFromNormals< 4 >( { 0, 0, 0, 1 }, 3 );
   testEntityDimensionFromNormals< 4 >( { 0, 0, 1, 0 }, 3 );
   testEntityDimensionFromNormals< 4 >( { 0, 1, 0, 0 }, 3 );
   testEntityDimensionFromNormals< 4 >( { 1, 0, 0, 0 }, 3 );

   testEntityDimensionFromNormals< 4 >( { 0, 0, 0, 0 }, 4 );
}

template<int GridDimension >
void testGetOrientationIndex( const typename TNL::Meshes::GridEntitiesOrientations<GridDimension>::NormalsType& normals,
                              int expectation ) {
   TNL::Meshes::GridEntitiesOrientations<GridDimension> entitiesOrientations;
   auto orientationIdx = entitiesOrientations.getOrientationIndex( normals );

   EXPECT_EQ( orientationIdx, expectation ) << " Grid Dimension: " << GridDimension
                                            << " Normals:" << normals;
}

TEST(GridEntitiesOrientationSuite, GetOrientationIndexTest_1D) {
   testGetOrientationIndex< 1 >( { 1 }, 0 );

   testGetOrientationIndex< 1 >( { 0 }, 0 );
}

TEST(GridEntitiesOrientationSuite, GetOrientationIndexTest_2D ) {
   testGetOrientationIndex< 2 >( { 1, 1 }, 0 );

   testGetOrientationIndex< 2 >( { 0, 1 }, 0 );
   testGetOrientationIndex< 2 >( { 1, 0 }, 1 );

   testGetOrientationIndex< 2 >( { 0, 0 }, 0 );
}

TEST(GridEntitiesOrientationSuite, GetOrientationIndexTest_3D ) {
   testGetOrientationIndex< 3 >( { 1, 1, 1 }, 0 );

   testGetOrientationIndex< 3 >( { 0, 1, 1 }, 0 );
   testGetOrientationIndex< 3 >( { 1, 0, 1 }, 1 );
   testGetOrientationIndex< 3 >( { 1, 1, 0 }, 2 );

   testGetOrientationIndex< 3 >( { 0, 0, 1 }, 0 );
   testGetOrientationIndex< 3 >( { 0, 1, 0 }, 1 );
   testGetOrientationIndex< 3 >( { 1, 0, 0 }, 2 );

   testGetOrientationIndex< 3 >( { 0, 0, 0 }, 0 );
}

TEST(GridEntitiesOrientationSuite, GetOrientationIndexTest_4D) {
   testGetOrientationIndex< 4 >( { 1, 1, 1, 1 }, 0 );

   testGetOrientationIndex< 4 >( { 0, 1, 1, 1 }, 0 );
   testGetOrientationIndex< 4 >( { 1, 0, 1, 1 }, 1 );
   testGetOrientationIndex< 4 >( { 1, 1, 0, 1 }, 2 );
   testGetOrientationIndex< 4 >( { 1, 1, 1, 0 }, 3 );

   testGetOrientationIndex< 4 >( { 0, 0, 1, 1 }, 0 );
   testGetOrientationIndex< 4 >( { 0, 1, 0, 1 }, 1 );
   testGetOrientationIndex< 4 >( { 0, 1, 1, 0 }, 2 );
   testGetOrientationIndex< 4 >( { 1, 0, 0, 1 }, 3 );
   testGetOrientationIndex< 4 >( { 1, 0, 1, 0 }, 4 );
   testGetOrientationIndex< 4 >( { 1, 1, 0, 0 }, 5 );

   testGetOrientationIndex< 4 >( { 0, 0, 0, 1 }, 0 );
   testGetOrientationIndex< 4 >( { 0, 0, 1, 0 }, 1 );
   testGetOrientationIndex< 4 >( { 0, 1, 0, 0 }, 2 );
   testGetOrientationIndex< 4 >( { 1, 0, 0, 0 }, 3 );

   testGetOrientationIndex< 4 >( { 0, 0, 0, 0 }, 0 );
}

template<int GridDimension >
void testGetTotalOrientationIndex( const typename TNL::Meshes::GridEntitiesOrientations<GridDimension>::NormalsType& normals,
                                   int expectation ) {
   TNL::Meshes::GridEntitiesOrientations<GridDimension> entitiesOrientations;
   auto totalOrientationIdx = entitiesOrientations.getTotalOrientationIndex( normals );

   EXPECT_EQ( totalOrientationIdx, expectation ) << " Grid Dimension: " << GridDimension
                                            << " Normals:" << normals;
}

TEST(GridEntitiesOrientationSuite, GetTotalOrientationIndexTest_1D) {
   testGetTotalOrientationIndex< 1 >( { 1 }, 0 );

   testGetTotalOrientationIndex< 1 >( { 0 }, 1 );
}

TEST(GridEntitiesOrientationSuite, GetTotalOrientationIndexTest_2D ) {
   testGetTotalOrientationIndex< 2 >( { 1, 1 }, 0 );

   testGetTotalOrientationIndex< 2 >( { 0, 1 }, 1 );
   testGetTotalOrientationIndex< 2 >( { 1, 0 }, 2 );

   testGetTotalOrientationIndex< 2 >( { 0, 0 }, 3 );
}

TEST(GridEntitiesOrientationSuite, GetTotalOrientationIndexTest_3D ) {
   testGetTotalOrientationIndex< 3 >( { 1, 1, 1 }, 0 );

   testGetTotalOrientationIndex< 3 >( { 0, 1, 1 }, 1 );
   testGetTotalOrientationIndex< 3 >( { 1, 0, 1 }, 2 );
   testGetTotalOrientationIndex< 3 >( { 1, 1, 0 }, 3 );

   testGetTotalOrientationIndex< 3 >( { 0, 0, 1 }, 4 );
   testGetTotalOrientationIndex< 3 >( { 0, 1, 0 }, 5 );
   testGetTotalOrientationIndex< 3 >( { 1, 0, 0 }, 6 );

   testGetTotalOrientationIndex< 3 >( { 0, 0, 0 }, 7 );
}

TEST(GridEntitiesOrientationSuite, GetTotalOrientationIndexTest_4D) {
   testGetTotalOrientationIndex< 4 >( { 1, 1, 1, 1 }, 0 );

   testGetTotalOrientationIndex< 4 >( { 0, 1, 1, 1 },  1 );
   testGetTotalOrientationIndex< 4 >( { 1, 0, 1, 1 },  2 );
   testGetTotalOrientationIndex< 4 >( { 1, 1, 0, 1 },  3 );
   testGetTotalOrientationIndex< 4 >( { 1, 1, 1, 0 },  4 );

   testGetTotalOrientationIndex< 4 >( { 0, 0, 1, 1 },  5 );
   testGetTotalOrientationIndex< 4 >( { 0, 1, 0, 1 },  6 );
   testGetTotalOrientationIndex< 4 >( { 0, 1, 1, 0 },  7 );
   testGetTotalOrientationIndex< 4 >( { 1, 0, 0, 1 },  8 );
   testGetTotalOrientationIndex< 4 >( { 1, 0, 1, 0 },  9 );
   testGetTotalOrientationIndex< 4 >( { 1, 1, 0, 0 }, 10 );

   testGetTotalOrientationIndex< 4 >( { 0, 0, 0, 1 }, 11 );
   testGetTotalOrientationIndex< 4 >( { 0, 0, 1, 0 }, 12 );
   testGetTotalOrientationIndex< 4 >( { 0, 1, 0, 0 }, 13 );
   testGetTotalOrientationIndex< 4 >( { 1, 0, 0, 0 }, 14 );

   testGetTotalOrientationIndex< 4 >( { 0, 0, 0, 0 }, 15 );
}

#include "../../main.h"

#endif

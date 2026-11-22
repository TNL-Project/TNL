add_subdirectory(Accessors)
add_subdirectory(Indexing)
add_subdirectory(Traverse)
add_subdirectory(Neighbours)
add_subdirectory(Utils)

tnl_add_test_executables(GridEntitiesOrientationsTest SOURCES GridEntitiesOrientationsTest.cpp BACKENDS HOST)

tnl_add_test_executables(GridWritersTest SOURCES GridWritersTest.cpp BACKENDS HOST)

tnl_add_test_executables(GridBoundaryTraverseTest SOURCES GridBoundaryTraverseTest.cpp BACKENDS HOST)

tnl_add_test_executables(GridAdjacencyTest SOURCES GridAdjacencyTest.cpp BACKENDS HOST)

tnl_add_test_executables(GridTraverseDeviceTest SOURCES GridTraverseDeviceTest.h BACKENDS CUDA HIP HOST SINGLE)

tnl_add_test_executables(GridReduceTest SOURCES GridReduceTest.h BACKENDS CUDA HIP HOST SINGLE)

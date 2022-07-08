add_subdirectory(tnl-quickstart)

# Mesh tools: will link to zlib and tinyxml2
set( MESH_TOOLS
         tnl-init
         tnl-diff
         tnl-decompose-grid
         tnl-grid-to-mesh
         tnl-mesh-converter
         tnl-mesh-info
         tnl-triangulate-mesh
         tnl-planar-correct-mesh
         tnl-refine-mesh
         tnl-game-of-life
         tnl-reorder-mesh
         tnl-plot-mesh-ordering
         tnl-test-distributed-mesh
)
set( CUDA_MESH_TOOLS )
if( BUILD_CUDA )
   set( CUDA_MESH_TOOLS  ${CUDA_MESH_TOOLS} tnl-test-distributed-mesh )
endif()

# MPI tools: will link to MPI
set( MPI_TOOLS
         tnl-init
         tnl-game-of-life
         tnl-test-distributed-mesh
)

foreach( target IN ITEMS ${MESH_TOOLS} )
   add_executable( ${target} ${target}.cpp )
endforeach()
foreach( target IN ITEMS ${CUDA_MESH_TOOLS} )
   cuda_add_executable( ${target}-cuda ${target}.cu )
   set( MESH_TOOLS  ${MESH_TOOLS} ${target}-cuda )
   set( MPI_TOOLS  ${MPI_TOOLS} ${target}-cuda )
endforeach()

find_package( ZLIB )
find_package( tinyxml2 QUIET )
foreach( target IN ITEMS ${MESH_TOOLS} )
   if( ZLIB_FOUND )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_ZLIB" )
      target_include_directories( ${target} PUBLIC ${ZLIB_INCLUDE_DIRS} )
      target_link_libraries( ${target} ${ZLIB_LIBRARIES} )
   endif()
   if( tinyxml2_FOUND )
      target_compile_definitions( ${target} PUBLIC "-DHAVE_TINYXML2" )
      target_link_libraries( ${target} tinyxml2::tinyxml2 )
   endif()
   install( TARGETS ${target} DESTINATION bin )
endforeach()

if( BUILD_MPI )
   foreach( target IN ITEMS ${MPI_TOOLS} )
      # enable MPI support in TNL
      target_compile_definitions( ${target} PUBLIC "-DHAVE_MPI" )
      # add MPI to the target: https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
      target_link_libraries( ${target} MPI::MPI_CXX )
   endforeach()
endif()

find_package( METIS QUIET )
if( METIS_FOUND )
   add_executable( tnl-decompose-mesh tnl-decompose-mesh.cpp )
   target_include_directories( tnl-decompose-mesh PUBLIC ${METIS_INCLUDE_DIRS} )
   target_link_libraries( tnl-decompose-mesh ${METIS_LIBRARIES} )
   if( ZLIB_FOUND )
      target_compile_definitions( tnl-decompose-mesh PUBLIC "-DHAVE_ZLIB" )
      target_include_directories( tnl-decompose-mesh PUBLIC ${ZLIB_INCLUDE_DIRS} )
      target_link_libraries( tnl-decompose-mesh ${ZLIB_LIBRARIES} )
   endif()
   if( tinyxml2_FOUND )
      target_compile_definitions( tnl-decompose-mesh PUBLIC "-DHAVE_TINYXML2" )
      target_link_libraries( tnl-decompose-mesh tinyxml2::tinyxml2 )
   endif()
   install( TARGETS tnl-decompose-mesh DESTINATION bin )
endif()

set( CGAL_DATA_DIR "/" )  # to disable stupid warning: https://github.com/CGAL/cgal/pull/6649
find_package( CGAL )
if( CGAL_FOUND )
   option( CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE "disable warning" ON )

   target_compile_definitions( tnl-reorder-mesh PUBLIC "-DHAVE_CGAL" )
   target_link_libraries( tnl-reorder-mesh CGAL::CGAL )

   target_compile_definitions( tnl-plot-mesh-ordering PUBLIC "-DHAVE_CGAL" )
   target_link_libraries( tnl-plot-mesh-ordering CGAL::CGAL )
endif()


add_executable( tnl-grid-setup tnl-grid-setup.cpp )

add_executable( tnl-image-converter tnl-image-converter.cpp )
if( PNG_FOUND )
   target_link_libraries( tnl-image-converter ${PNG_LIBRARIES} )
endif()
if( JPEG_FOUND )
   target_link_libraries( tnl-image-converter ${JPEG_LIBRARIES} )
endif()

add_executable( tnl-dicom-reader tnl-dicom-reader.cpp )
if( DCMTK_FOUND )
   target_link_libraries( tnl-dicom-reader ${DCMTK_LIBRARIES} )
endif()

IF( BUILD_CUDA )
   cuda_add_executable( tnl-cuda-arch tnl-cuda-arch.cu )
   install( TARGETS tnl-cuda-arch DESTINATION bin )
ENDIF()

install( TARGETS tnl-grid-setup
                 tnl-image-converter
                 tnl-dicom-reader
         DESTINATION bin )

install( PROGRAMS tnl-err2eoc
                  tnl-benchmark-to-html.py
                  tnl-log-to-html.py
         DESTINATION bin )

include("../../CMakeSupport.cmake")

SET(SOME_ARGS "")

generate_executables(Grid3D_0DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)
generate_executables(Grid3D_1DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)
generate_executables(Grid3D_2DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)
generate_executables(Grid3D_3DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)

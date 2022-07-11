include("../../CMakeSupport.cmake")

SET(SOME_ARGS "")

generate_executables(Grid2D_0DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)
generate_executables(Grid2D_1DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)
generate_executables(Grid2D_2DTest.h "Neighbours" "../../../../main.h" SOME_ARGS)

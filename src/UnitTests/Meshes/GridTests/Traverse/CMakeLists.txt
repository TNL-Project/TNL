include("../CMakeSupport.cmake")

SET(SOME_ARGS "")

generate_executables(Grid1DTest.h "Traversers" "../../../main.h" SOME_ARGS)
generate_executables(Grid2DTest.h "Traversers" "../../../main.h" SOME_ARGS)
#generate_executables(Grid2DTest.h "Accessors" SOME_ARGS)

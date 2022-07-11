include("../CMakeSupport.cmake")

SET(SOME_ARGS "")

generate_executables(Grid1DTest.h "Accessors" "../../../main.h" SOME_ARGS)
generate_executables(Grid2DTest.h "Accessors" "../../../main.h" SOME_ARGS)
generate_executables(Grid3DTest.h "Accessors" "../../../main.h" SOME_ARGS)

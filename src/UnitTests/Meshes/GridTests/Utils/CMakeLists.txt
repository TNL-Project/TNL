include("../CMakeSupport.cmake")

SET(SOME_ARGS "")

generate_executables(GridBasisTest.h "Utils" "../../../main.h" SOME_ARGS)
generate_executables(GridTemplatesTest.h "Utils" "../../../main.h" SOME_ARGS)

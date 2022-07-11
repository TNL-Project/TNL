include("../CMakeSupport.cmake")

SET(SOME_ARGS "")

generate_executables(GridBasisTest.h "Utils" "../../../main.h" SOME_ARGS)
generate_executables(GridTemplatesTest.h "Utils" "../../../main.h" SOME_ARGS)
generate_executables(EntityDataStoreTest.h "Utils" "../../../main.h" SOME_ARGS)
generate_executables(CoordinateIteratorTest.h "Utils" "../../../main.h" SOME_ARGS)
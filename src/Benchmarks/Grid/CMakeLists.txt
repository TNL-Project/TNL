
function(generate_executables HEADER SUFFIX MAIN_FILE_PATH ARGS)
get_filename_component(MODULE_NAME ${HEADER} NAME_WE)

if (${BUILD_CUDA})
   SET(SUPPORT_FILENAME ${MODULE_NAME}.cu)
else()
   SET(SUPPORT_FILENAME ${MODULE_NAME}.cpp)
endif()

file(WRITE ${SUPPORT_FILENAME}
"
#include \"${MAIN_FILE_PATH}\"
")

# Targets are separated by suffix provided
set(MODULE_NAME ${MODULE_NAME}_${SUFFIX})

if (${BUILD_CUDA})
   CUDA_ADD_EXECUTABLE(${MODULE_NAME} ${HEADER} ${SUPPORT_FILENAME} OPTIONS ${CUDA_TESTS_FLAGS})
   TARGET_LINK_LIBRARIES(${MODULE_NAME} ${TESTS_LIBRARIES})
else()
   ADD_EXECUTABLE(${MODULE_NAME} ${HEADER} ${SUPPORT_FILENAME})
   TARGET_COMPILE_OPTIONS(${MODULE_NAME} PRIVATE ${CXX_TESTS_FLAGS})
endif()

endfunction()

SET(SOME_ARGS "")

generate_executables(Grid1D.h "OperationBenchmark" "Grid1D.h" SOME_ARGS)
generate_executables(Grid2D.h "OperationBenchmark" "Grid2D.h" SOME_ARGS)
generate_executables(Grid3D.h "OperationBenchmark" "Grid3D.h" SOME_ARGS)

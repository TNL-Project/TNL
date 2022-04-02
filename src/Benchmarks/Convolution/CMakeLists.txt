
function(generate_cuda_executable PREFIX DIMENSION TEMPLATE KERNEL_HEADER)

get_filename_component(MODULE_NAME ${KERNEL_HEADER} NAME_WE)
get_filename_component(TEMPLATE_NAME ${TEMPLATE} NAME_WE)

if (${BUILD_CUDA})
   SET(SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/generated/${MODULE_NAME}_${DIMENSION}_${TEMPLATE_NAME}.cu")

   FILE(READ ${TEMPLATE} TEMPLATE_CONTENT)

   STRING(REGEX REPLACE "DIMENSION_VALUE" ${DIMENSION} TEMPLATE_CONTENT "${TEMPLATE_CONTENT}")

   FILE(WRITE ${SOURCE_FILE} "${TEMPLATE_CONTENT}")

   SET(EXECUTABLE_NAME "${PREFIX}_${DIMENSION}_${MODULE_NAME}")

   CUDA_ADD_EXECUTABLE(${EXECUTABLE_NAME} ${SOURCE_FILE})
else()
   MESSAGE(WARNING "Convolutions are not supported on CPU")
endif()

endfunction()

GENERATE_CUDA_EXECUTABLE("Convolution" 1 "templates/main_solver.h" "kernels/naive.h")
GENERATE_CUDA_EXECUTABLE("Convolution" 2 "templates/main_solver.h" "kernels/naive.h")

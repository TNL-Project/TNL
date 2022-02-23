
function(generate_executables HEADER ARGS)
   get_filename_component(MODULE_NAME ${HEADER} NAME_WE)

   if (${BUILD_CUDA})
      SET(SUPPORT_FILENAME ${MODULE_NAME}.cu)
   else()
      SET(SUPPORT_FILENAME ${MODULE_NAME}.cpp)
   endif()

   file(WRITE ${SUPPORT_FILENAME}
"
#include \"${HEADER}\"
#include \"../../main.h\"
")

   if (${BUILD_CUDA})
      CUDA_ADD_EXECUTABLE(${MODULE_NAME} ${HEADER} ${SUPPORT_FILENAME} OPTIONS ${CUDA_TESTS_FLAGS})
      TARGET_LINK_LIBRARIES(${MODULE_NAME} ${TESTS_LIBRARIES})
   else()
      ADD_EXECUTABLE(${MODULE_NAME} ${HEADER} ${SUPPORT_FILENAME})
      TARGET_COMPILE_OPTIONS(${MODULE_NAME} PRIVATE ${CXX_TESTS_FLAGS})
   endif()

   TARGET_LINK_LIBRARIES(${MODULE_NAME} ${TESTS_LIBRARIES} )
   target_link_options(${MODULE_NAME} PRIVATE ${TESTS_LINKER_FLAGS})
   ADD_TEST(${MODULE_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${MODULE_NAME}${CMAKE_EXECUTABLE_SUFFIX} )
endfunction()

SET(SOME_ARGS "")

generate_executables(Grid1DTest.h SOME_ARGS)
generate_executables(Grid2DTest.h SOME_ARGS)
#generate_executables(Grid3DTest.h SOME_ARGS)

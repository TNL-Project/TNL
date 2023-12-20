set( PROJECT_NAME  smallEigenIterationRandomDense bigEigenIterationRandomDense bigEigenIterationRandomSparse smallEigenIterationShiftRandomDense eigenIterationFileSparse eigenQR)

if(TNL_BUILD_CUDA)
   foreach( target IN ITEMS ${PROJECT_NAME} )
      add_executable( ${target} ${target}.cu )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CUDA )
   endforeach()
else()
   foreach( target IN ITEMS ${PROJECT_NAME} )
      add_executable( ${target} ${target}.cpp )
      target_link_libraries( ${target} PUBLIC TNL::TNL_CXX )
   endforeach()
endif()


# add_executable(TNL_example_test1 main.cpp)
# target_link_libraries(TNL_example_test1 PUBLIC TNL::TNL_CXX )

# if(TNL_BUILD_CUDA)
#    add_executable(TNL_example_test1_cuda main.cu)
#    target_link_libraries(TNL_example_test1_cuda PUBLIC TNL::TNL_CUDA )
# endif()

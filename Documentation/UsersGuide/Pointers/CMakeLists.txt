add_executable(UniquePointerHostExample UniquePointerHostExample.cpp)
target_link_libraries(UniquePointerHostExample PUBLIC TNL::TNL_CXX)
add_custom_command(
    COMMAND UniquePointerHostExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out
    OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out DEPENDS UniquePointerHostExample
)

add_custom_target(RunUGPointersExamples ALL DEPENDS ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out)

# add the dependency to the main target
add_dependencies(run-doc-examples RunUGPointersExamples)

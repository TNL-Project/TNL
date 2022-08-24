ADD_EXECUTABLE( UniquePointerHostExample UniquePointerHostExample.cpp )
target_link_libraries( UniquePointerHostExample PUBLIC TNL::TNL_CXX )
ADD_CUSTOM_COMMAND( COMMAND UniquePointerHostExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out
                    OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out
                    DEPENDS UniquePointerHostExample )

ADD_CUSTOM_TARGET( RunTutorialsPointersExamples ALL DEPENDS
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out
)

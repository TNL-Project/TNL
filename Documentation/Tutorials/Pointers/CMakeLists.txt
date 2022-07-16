ADD_EXECUTABLE( UniquePointerHostExample UniquePointerHostExample.cpp )
ADD_CUSTOM_COMMAND( COMMAND UniquePointerHostExample > ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out OUTPUT ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out )

ADD_CUSTOM_TARGET( TutorialsPointers ALL DEPENDS
   ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}/UniquePointerHostExample.out
)

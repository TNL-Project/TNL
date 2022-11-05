if( TNL_BUILD_DOC )
   # create the directory where the output of running code examples will be collected
   set( TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH "${CMAKE_CURRENT_BINARY_DIR}/output_snippets" )
   file( MAKE_DIRECTORY ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH} )

   # add a target that will collect all commands to run the code examples
   add_custom_target( run-doc-examples ALL )

   # add subdirectories that contain code examples included in the documentation
   add_subdirectory( Examples )
   add_subdirectory( Tutorials )

   # extract the version string during the configure step
   execute_process( COMMAND ${CMAKE_SOURCE_DIR}/scripts/get-version
                    OUTPUT_VARIABLE TNL_DOC_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE )

   # check if Doxygen is installed
   find_package( Doxygen REQUIRED )

   # collect source file dependencies for the doxygen command
   file( GLOB_RECURSE TNL_DOC_INPUT_FILES
         ${CMAKE_SOURCE_DIR}/src/TNL/*
         ${CMAKE_SOURCE_DIR}/Documentation/Examples/*
         ${CMAKE_SOURCE_DIR}/Documentation/Tutorials/*
   )

   # the documentation cannot be built without warnings when CUDA is disabled
   # (e.g. output of some examples will be missing)
   if( DEFINED ENV{GITLAB_CI} OR TNL_BUILD_CUDA )
      set( DOXYGEN_WARN_AS_ERROR "YES" )
   else()
      set( DOXYGEN_WARN_AS_ERROR "NO" )
   endif()

   # add custom target to generate the documentation during the build
   set( DOXYGEN_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp )
   add_custom_command(
      COMMAND ${CMAKE_COMMAND} -E env PROJECT_NUMBER="version ${TNL_DOC_VERSION}"
                                      OUTPUT_SNIPPETS_PATH="${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}"
                                      OUTPUT_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}"
                                      WARN_AS_ERROR="${DOXYGEN_WARN_AS_ERROR}"
              ${DOXYGEN_EXECUTABLE}
      COMMAND cmake -E touch "${DOXYGEN_STAMP_FILE}"
      OUTPUT ${DOXYGEN_STAMP_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating documentation with Doxygen" )

   # add custom target to run doxygen (depending on the stamp file allows to
   # track the dependencies on the input files)
   add_custom_target( doxygen ALL DEPENDS ${DOXYGEN_STAMP_FILE} )

   # add dependency on the run-doc-examples target to the doxygen target
   add_dependencies( doxygen run-doc-examples )

   # remove existing files under the install prefix
   install( CODE "MESSAGE( \"-- Removing existing documentation: ${CMAKE_INSTALL_PREFIX}/share/doc/tnl/\")" )
   install( CODE "file( REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/share/doc/tnl\" )" )

   # install the newly built documentation
   install( CODE "MESSAGE( \"-- Installing documentation: ${CMAKE_INSTALL_PREFIX}/share/doc/tnl/html/\")" )
   install( DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html" DESTINATION "share/doc/tnl" MESSAGE_NEVER )
endif()

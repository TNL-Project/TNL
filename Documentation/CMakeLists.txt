# create the directory where the output of running code examples will be collected
set(TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH "${CMAKE_CURRENT_BINARY_DIR}/output_snippets")
file(MAKE_DIRECTORY ${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH})

# add a target that will collect all commands to run the code examples
add_custom_target(run-doc-examples ALL)

# add subdirectories that contain code examples included in the documentation
add_subdirectory(Examples)
add_subdirectory(UsersGuide)

# check if Doxygen is installed
find_package(Doxygen)

if(DOXYGEN_FOUND)
    # warn if Doxygen version does not match what we expect
    set(TNL_EXPECTED_DOXYGEN_VERSION 1.11.0)
    if(NOT ${DOXYGEN_VERSION} VERSION_GREATER_EQUAL ${TNL_EXPECTED_DOXYGEN_VERSION})
        message(
            WARNING
            "Found Doxygen version ${DOXYGEN_VERSION}, which does not match the expected version "
            "${TNL_EXPECTED_DOXYGEN_VERSION}. The generated documentation output may be incorrect due to "
            "various Doxygen quirks and bugs."
        )
    endif()

    # extract the version string during the configure step
    execute_process(
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/get-version
        OUTPUT_VARIABLE TNL_DOC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # the documentation cannot be built without warnings when CUDA is disabled
    # (e.g. output of some examples will be missing)
    if(DEFINED ENV{GITLAB_CI} OR (TNL_USE_CI_FLAGS AND TNL_BUILD_CUDA))
        set(DOXYGEN_WARN_AS_ERROR "YES")
    else()
        set(DOXYGEN_WARN_AS_ERROR "NO")
    endif()

    if(TNL_DOXYGEN_USE_CLANG)
        set(DOXYGEN_CLANG_ASSISTED_PARSING "YES")
        # Check --print-resource-dir to fix libclang being unable find certain resources like `stddef.h`
        # https://github.com/googleapis/google-cloud-cpp/commit/765e427e07eb53e726d25066e2b65f4b17911b99
        execute_process(
            COMMAND clang --print-resource-dir
            RESULT_VARIABLE COMMAND_RETURN_CODE
            OUTPUT_VARIABLE CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(COMMAND_RETURN_CODE EQUAL 0)
            set(DOXYGEN_CLANG_OPTIONS "-resource-dir=${CLANG_RESOURCE_DIR} -Wno-pragma-once-outside-header")
        else()
            set(DOXYGEN_CLANG_OPTIONS "")
            message(WARNING "DOXYGEN_CLANG_OPTIONS not set: check that clang is properly installed")
        endif()
        message(STATUS "DOXYGEN_CLANG_OPTIONS = ${DOXYGEN_CLANG_OPTIONS}")
    else()
        set(DOXYGEN_CLANG_ASSISTED_PARSING "NO")
        set(DOXYGEN_CLANG_OPTIONS "")
    endif()

    # collect source file dependencies for the doxygen command
    file(GLOB_RECURSE TNL_DOC_INPUT_FILES ${CMAKE_SOURCE_DIR}/src/TNL/* ${CMAKE_SOURCE_DIR}/Documentation/*)

    # add custom target to generate the documentation during the build
    set(DOXYGEN_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp)
    add_custom_command(
        COMMAND
        ${CMAKE_COMMAND}
        -E
        env
        PROJECT_NUMBER="version ${TNL_DOC_VERSION}"
        OUTPUT_SNIPPETS_PATH="${TNL_DOCUMENTATION_OUTPUT_SNIPPETS_PATH}"
        OUTPUT_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}"
        WARN_AS_ERROR="${DOXYGEN_WARN_AS_ERROR}"
        CLANG_ASSISTED_PARSING="${DOXYGEN_CLANG_ASSISTED_PARSING}"
        CLANG_OPTIONS="${DOXYGEN_CLANG_OPTIONS}"
        CLANG_DATABASE_PATH="${CMAKE_CURRENT_BINARY_DIR}"
        ${DOXYGEN_EXECUTABLE}
        COMMAND
        cmake
        -E
        touch
        "${DOXYGEN_STAMP_FILE}"
        DEPENDS
        ${TNL_DOC_INPUT_FILES}
        OUTPUT
        ${DOXYGEN_STAMP_FILE}
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT
        "Generating documentation with Doxygen"
    )

    # add custom target to run doxygen (depending on the stamp file allows to
    # track the dependencies on the input files)
    add_custom_target(doxygen ALL DEPENDS ${DOXYGEN_STAMP_FILE})

    # add dependency on the run-doc-examples target to the doxygen target
    add_dependencies(doxygen run-doc-examples)

    # remove existing files under the install prefix
    install(
        CODE "MESSAGE( \"-- Removing existing documentation: ${CMAKE_INSTALL_PREFIX}/share/doc/tnl/\")"
        COMPONENT documentation
    )
    install(CODE "file( REMOVE_RECURSE \"${CMAKE_INSTALL_PREFIX}/share/doc/tnl\" )" COMPONENT documentation)

    # install the newly built documentation
    install(
        CODE "MESSAGE( \"-- Installing documentation: ${CMAKE_INSTALL_PREFIX}/share/doc/tnl/html/\")"
        COMPONENT documentation
    )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html" DESTINATION "share/doc/tnl" MESSAGE_NEVER COMPONENT documentation)
else()
    message(WARNING "Doxygen was not found, the documentation will not be built.")
endif()

# collect targets defined in this directory
include(get_all_targets)
get_all_targets(TNL_DOCUMENTATION_TARGETS ${CMAKE_CURRENT_LIST_DIR})

# add utility target for this directory
add_custom_target(documentation)
add_dependencies(documentation ${TNL_DOCUMENTATION_TARGETS})

# select one default GPU architecture to avoid building for all GPU architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(CMAKE_CUDA_ARCHITECTURES "native")
    else()
        # CMake's native GPU arch detection does not work with Clang
        execute_process(
            COMMAND "nvidia-smi" "--query-gpu=compute_cap" "--format=csv,noheader"
            RESULT_VARIABLE _status
            OUTPUT_VARIABLE _stdout
            ERROR_VARIABLE _stderr
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(${_status} EQUAL 0)
            string(REPLACE "." "" _stdout ${_stdout})
            string(REGEX REPLACE "[ \t\r\n]+" ";" CMAKE_CUDA_ARCHITECTURES ${_stdout})
            list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
        else()
            message(WARNING "Failed to detect native GPU architecture using nvidia-smi: ${_stderr}")
        endif()
    endif()
    message("-- Selected CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
endif()

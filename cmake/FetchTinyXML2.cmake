include(FetchContent)
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG 10.0.0
    FIND_PACKAGE_ARGS NAMES tinyxml2 SYSTEM
)
set(tinyxml2_BUILD_TESTING OFF)
FetchContent_MakeAvailable(tinyxml2)

#!/usr/bin/env -S just --working-directory . --justfile

set dotenv-load := true

# Flags for tools that do not respect the FORCE_COLOR environment variable

color := if env("FORCE_COLOR", "") == "1" { "--color" } else { "" }
color_always := if env("FORCE_COLOR", "") == "1" { "--color always" } else { "" }

# NOTE: The configure recipe is intended to be replaceable - users can run cmake manually
#       with their desired options and then use other recipes like `build` or `check`.

# Configures the project for building with cmake using options from the .env file and environment variables
configure:
    #!/usr/bin/env bash
    set -euo pipefail

    # Enable TNL_USE_CI_FLAGS when running in a CI environment
    if [[ "${CI:-false}" == "true" ]]; then
        # shellcheck disable=SC2034
        TNL_USE_CI_FLAGS=ON
    fi

    # Collect all variables that start with CMAKE_ or TNL_ (Note that `env` sees
    # only exported variables, but `set` is a built-in command that sees all
    # variables defined in the .env file.)
    declare -a cmake_options
    for var in $(set | grep -E '^CMAKE_|^TNL_'); do
        cmake_options+=(-D "$var")
    done

    just _ensure-command cmake
    cmake -B "$BUILD_DIR" -S . "${cmake_options[@]}"

# Lists available build targets
list-build-targets:
    just _ensure-cmake-configured
    cmake --build build --target help 2>/dev/null | grep ": phony" | grep -v "/" | sed 's/: phony//'

# Builds the project using cmake (this is the default recipe)
[default]
build +targets="all":
    just _ensure-cmake-configured
    cmake --build build --target {{ targets }}

# Installs the project using cmake (valid components: headers, benchmarks, documentation, examples, tools, or all)
install +components="headers":
    #!/usr/bin/env bash
    set -euo pipefail

    # Validate components argument as a space-separated string
    # shellcheck disable=SC2043
    for component in {{ components }}; do
        if [[ ! "$component" =~ ^(headers|benchmarks|documentation|examples|tools|all)$ ]]; then
            echo "Invalid component: $component" >&2
            exit 1
        fi
    done

    # All components except headers correspond to build targets, so build them
    # shellcheck disable=SC2050
    if [[ "{{ components }}" != "headers" ]]; then
        just build {{ replace(components, "headers", "") }}
    fi

    # Installing all components must be done without the --component flag
    # shellcheck disable=SC2050
    if [[ "{{ components }}" == "all" ]]; then
        cmake --install build
    else
        # shellcheck disable=SC2043
        for component in {{ components }}; do
            cmake --install build --component "$component"
        done
    fi

# Runs all tests using ctest
test: (build "tests")
    just _ensure-cmake-configured
    just _ensure-command ctest
    ctest --preset all-tests --test-dir build

# Cleans the build directory
clean:
    rm -frv "$BUILD_DIR"

# Runs all checks
check: check-typos check-format check-recipes check-shellcheck check-python check-clang-tidy

# Checks for common spelling mistakes using typos
check-typos:
    just _ensure-command typos
    typos {{ color_always }} --sort

# Checks the code formatting using clang-format and ruff
check-format:
    just --unstable --fmt --check
    just _ensure-command clang-format
    ./scripts/run-clang-format.py {{ color_always }} --style file --exclude "src/TNL/3rdparty/*" --recursive Documentation src
    just _ensure-command gersemi
    gersemi {{ color }} --diff --check .
    just _ensure-command ruff
    ruff format --diff

# Reformats supported files using clang-format, gersemi, and ruff
format:
    just --unstable --fmt
    just _ensure-command clang-format
    ./scripts/run-clang-format.py {{ color_always }} --style file --in-place --exclude "src/TNL/3rdparty/*" --recursive Documentation src
    just _ensure-command gersemi
    gersemi {{ color }} --in-place .
    just _ensure-command ruff
    ruff format .

# Checks justfile recipe for shell issues using shellcheck
_check-recipe recipe:
    just _ensure-command grep shellcheck
    just -vv -n {{ recipe }} 2>&1 | grep -v '===> Running recipe' | shellcheck -

# Checks all justfile recipes with inline bash for shell issues using shellcheck
check-recipes:
    just _check-recipe '_ensure-command command'
    just _check-recipe 'check-shellcheck'
    just _check-recipe 'configure'
    just _check-recipe 'install'
    just _check-recipe 'create-gitlab-release'
    just _check-recipe 'release'

# Check shell scripts for code quality using shellcheck
check-shellcheck:
    #!/usr/bin/env bash
    set -euo pipefail

    # Find all bash files in the current git repo
    just _ensure-command git
    grep_args=(-l -E -e '^#! *(/usr)?/bin/(env )?(ba)?sh' -e '# *shellcheck shell=(ba)?sh)$')  # spellchecker:disable-line
    # Include untracked files
    grep_args+=(--untracked)
    shell_files_1=$(git grep "${grep_args[@]}")
    shell_files_2=$(git ls-files '**.sh' '**.bash' 2>/dev/null)

    # Read the file names as an array and make the list unique
    readarray -t shell_files < <(echo -e "${shell_files_1}\n${shell_files_2}" | sort | uniq)

    # Check shell files for issues
    just _ensure-command shellcheck
    shellcheck "${shell_files[@]}"

# Checks python scripts for code quality
check-python:
    just _ensure-command ruff
    ruff check
    # TODO:
    #just _ensure-command basedpyright
    #basedpyright
    #just _ensure-command mypy
    #mypy

# Checks the code using ruff
check-clang-tidy +target_paths="Documentation/.* src/Benchmarks/.* src/Examples/.* src/Tools/.*":
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-compile-commands-json

    # check that the preset does not involve the CUDA toolchain
    use_cuda=$(grep TNL_USE_CUDA:BOOL= "$BUILD_DIR"/CMakeCache.txt | cut -d'=' -f2)
    if [[ "$use_cuda" =~ ^(ON|YES|TRUE|1)$ ]]; then
        echo "clang-tidy check is not supported with CUDA. Reconfigure with TNL_USE_CUDA=OFF and try again." >&2
        return 1
    fi

    # check that the build type is Debug
    build_type=$(grep CMAKE_BUILD_TYPE:STRING= "$BUILD_DIR"/CMakeCache.txt | cut -d'=' -f2)
    if [[ "$build_type" != "Debug" ]]; then
        echo "clang-tidy should be run only in Debug build. Reconfigure with CMAKE_BUILD_TYPE=Debug and try again." >&2
        return 1
    fi

    # run-clang-tidy is weird compared to run-clang-format.py:
    # - clang-tidy is not executed on header files, but on source files
    # - the positional arguments are regexes filtering sources from the compile_commands.json
    # - the -header-filter option (or HeaderFilterRegex in the config) allows to filter header files
    # - "build/.*" must be excluded on run-clang-tidy command line as it
    #   includes targets generated by CMake's FetchContent module
    run-clang-tidy -quiet -p "$BUILD_DIR" {{ target_paths }}

# Ensures that one or more required commands are installed
_ensure-command +command:
    #!/usr/bin/env bash
    set -euo pipefail

    read -r -a commands <<< "{{ command }}"

    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" > /dev/null 2>&1 ; then
            printf "Couldn't find required executable '%s'\n" "$cmd" >&2
            exit 1
        fi
    done

# Ensures that the CMakeCache.txt file exists
_ensure-cmake-configured:
    test -f "$BUILD_DIR"/CMakeCache.txt || just configure

# Ensures that the compile_commands.json file exists
_ensure-compile-commands-json:
    # for some reason cmake creates compile_commands.json only on the second run
    test -f "$BUILD_DIR"/compile_commands.json || just configure
    test -f "$BUILD_DIR"/compile_commands.json || just configure

# Gets the current version of the project from the CMakeLists.txt file
get-current-version:
    # NOTE: parsing CMakeCache.txt would be easier, but it is not guaranteed to be up-to-date
    #grep CMAKE_PROJECT_VERSION:STATIC= "$BUILD_DIR"/CMakeCache.txt | cut -d'=' -f2
    # This command will:
    # - Look for lines that start with project(
    # - Continue until the matching closing parenthesis )
    # - Within that block, find any line containing VERSION
    # - Print the token immediately following VERSION
    awk '/project\s*\(/,/\)/ {if (/VERSION/) {for(i=1;i<=NF;i++) if($i=="VERSION") {print $(i+1); break}}}' CMakeLists.txt

# Creates a GitLab release using the glab CLI tool
create-gitlab-release:
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-command git

    # Check that we are on the main branch
    if [[ "$(git branch --show-current)" != "main" ]]; then
        printf "You are not on the main branch!\n" >&2
        exit 1
    fi

    # Pull the latest changes
    git pull --tags origin

    # Get the current version from the last git tag
    current_version="$(git tag --sort=version:refname | tail -n 1)"
    if [[ -z "$current_version" ]]; then
        printf "No current version found!\n" >&2
        exit 1
    fi

    # Get the previous version (if any)
    previous_version="$(git tag --sort=version:refname | tail -n 2 | head -n 1)"

    # Prepare initial release notes (can be edited manually on GitLab)
    release_notes="# Release notes - version $current_version\n\n"
    if [[ -n "$previous_version" ]]; then
        release_notes+="## Merge requests\n\n$(git log --pretty=format:"* %w(0,0,2)%b" --merges "$previous_version..$current_version")\n\n"
        release_notes+="## Detailed changes\n\n$(git log --pretty=format:"* %s (%H)" --no-merges "$previous_version..$current_version")\n\n"
    fi
    # Run through printf to interpret escapes such as \n
    release_notes="$(echo -e "$release_notes")"

    just _ensure-command glab

    printf "Creating GitLab release %s\n" "$current_version"
    glab release create "$current_version" --no-update --ref="$current_version" --name="$current_version" --notes="$release_notes"

# Creates a tag and pushes it (if it does exist yet) and creates a release for it
release:
    #!/usr/bin/env bash
    set -euo pipefail

    just _ensure-command git

    # Check that we are on the main branch
    if [[ "$(git branch --show-current)" != "main" ]]; then
        printf "You are not on the main branch!\n" >&2
        exit 1
    fi

    # Pull the latest changes
    git pull --tags origin

    # Get the current version of the project
    current_version="$(just get-current-version)"
    readonly current_version="$current_version"
    if [[ -z "$current_version" ]]; then
        printf "No current version found!\n" >&2
        exit 1
    fi

    # Check that the tag does not exist yet
    if [[ -n "$(git tag -l "$current_version")" ]]; then
        printf "The tag %s exists already!\n" "$current_version" >&2
        exit 1
    fi

    # Create a new tag and push it
    git push origin
    printf "Creating tag %s...\n" "$current_version"
    git tag -a "$current_version" -m "version $current_version"
    printf "Pushing tag %s...\n" "$current_version"
    git push origin refs/tags/"$current_version"

    # Create a release on GitLab
    just create-gitlab-release

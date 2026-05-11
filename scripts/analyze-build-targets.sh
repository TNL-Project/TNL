#!/bin/bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [TARGETS...]

Analyze maximum memory usage when building project targets.

Meta-targets (CTest, install, etc.) are automatically excluded from the target list.

Options:
    -h, --help              Show this help message
    -f, --filter PATTERN    Filter targets matching pattern (grep regex)
    -e, --exclude PATTERN   Exclude additional targets matching pattern (grep regex)
    -n, --dry-run           Show what would be built without actually building
    --clean                 Clean build directory before analysis (slower but more accurate)
    --parallel [N]          Run N builds in parallel using GNU parallel (default: all CPU cores)

Examples:
    $0                              # Analyze all targets (excluding meta-targets)
    $0 -f "Test$"                   # Analyze targets ending with 'Test'
    $0 -e "Cuda"                    # Exclude Cuda targets additionally
    $0 tnl-benchmark-blas tests     # Analyze specific targets
    $0 --parallel 8                 # Analyze with 8 parallel builds

Output:
    Results are saved to build/targets_analysis/ directory:
    - targets_summary_TIMESTAMP.csv  - CSV with target, max_rss_kb, elapsed_time_sec, binary_size_kb, is_cuda, host_functions, cuda_kernels
EOF
}

FILTER=""
EXCLUDE=""
DRY_RUN=false
CLEAN=false
PARALLEL_JOBS=1
USER_TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -f|--filter)
            if [[ $# -lt 2 ]] || [[ "$2" =~ ^- ]]; then
                echo "Error: --filter requires a pattern argument" >&2
                usage
                exit 1
            fi
            FILTER="$2"
            shift 2
            ;;
        -e|--exclude)
            if [[ $# -lt 2 ]] || [[ "$2" =~ ^- ]]; then
                echo "Error: --exclude requires a pattern argument" >&2
                usage
                exit 1
            fi
            EXCLUDE="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --parallel)
            if [[ $# -lt 2 ]] || [[ "$2" =~ ^- ]]; then
                PARALLEL_JOBS=$(nproc)
                shift
            else
                PARALLEL_JOBS="$2"
                shift 2
            fi
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            USER_TARGETS+=("$1")
            shift
            ;;
    esac
done

if [[ $PARALLEL_JOBS -gt 1 ]] && ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is required for --parallel but not found." >&2
    echo "Install with: sudo apt install parallel" >&2
    exit 1
fi

OUTPUT_DIR="$PWD/build/targets_analysis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/targets_summary_${TIMESTAMP}.csv"
RESULTS_DIR="$OUTPUT_DIR/results_${TIMESTAMP}"

DEFAULT_EXCLUDE="^(Experimental|Nightly|Continuous|libgtest|libgmock|gtest|gmock)|^(test|edit_cache|rebuild_cache|list_install_components|install|benchmarks|tests|tools|matrix-tests|non-matrix-tests|examples|documentation|run-doc-examples|doxygen)$"

if [[ ${#USER_TARGETS[@]} -gt 0 ]]; then
    TARGETS="${USER_TARGETS[*]}"
else
    TARGETS=$(just list-build-targets 2>/dev/null | grep -Ev "$DEFAULT_EXCLUDE")
fi

if [[ -n "$FILTER" ]]; then
    TARGETS=$(echo "$TARGETS" | grep -E "$FILTER")
fi

if [[ -n "$EXCLUDE" ]]; then
    TARGETS=$(echo "$TARGETS" | grep -Ev "$EXCLUDE")
fi

TARGETS_ARRAY=()
for target in $TARGETS; do
    TARGETS_ARRAY+=("$target")
done

TOTAL_TARGETS=${#TARGETS_ARRAY[@]}

if [[ $TOTAL_TARGETS -eq 0 ]]; then
    echo "No targets found matching criteria" >&2
    exit 1
fi

echo "Starting memory analysis at $(date)"
echo "========================================"
echo "Targets to analyze: $TOTAL_TARGETS"
if [[ -n "$FILTER" ]]; then
    echo "Filter: $FILTER"
fi
if [[ -n "$EXCLUDE" ]]; then
    echo "Exclude: $EXCLUDE"
fi
if [[ "$CLEAN" == true ]]; then
    echo "Clean mode: enabled"
fi
if [[ $PARALLEL_JOBS -gt 1 ]]; then
    echo "Parallel jobs: $PARALLEL_JOBS"
fi
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run - targets that would be analyzed:"
    for target in "${TARGETS_ARRAY[@]}"; do
        echo "  - $target"
    done
    exit 0
fi

if [[ "$CLEAN" == true ]]; then
    just build clean
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"
echo "target,max_rss_kb,elapsed_time_sec,binary_size_kb,is_cuda,host_functions,cuda_kernels" > "$SUMMARY_FILE"

analyze_target() {
    local target="$1"
    local RESULTS_DIR="$2"
    local RESULT_FILE="$RESULTS_DIR/$target.result"

    cd "$PWD" || exit 1

    local TIME_OUTPUT
    TIME_OUTPUT=$(mktemp)

    set +e
    command time --verbose just build "$target" > /dev/null 2>"$TIME_OUTPUT"
    local EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "FAILED" > "$RESULT_FILE"
        rm -f "$TIME_OUTPUT"
        return 0
    fi

    local MAX_RSS ELAPSED ELAPSED_SEC
    MAX_RSS=$(grep "Maximum resident set size" "$TIME_OUTPUT" 2>/dev/null | awk '{print $NF}' || echo "0")
    ELAPSED=$(grep "Elapsed (wall clock) time" "$TIME_OUTPUT" 2>/dev/null | sed 's/.*: //' || echo "0:00.00")

    if command -v bc &> /dev/null; then
        ELAPSED_SEC=$(echo "$ELAPSED" | awk -F: '{ if (NF == 2) { printf "%.2f", $1 * 60 + $2 } else { gsub(",", "."); print $1 } }')
    else
        ELAPSED_SEC=$(echo "$ELAPSED" | awk -F: '{ if (NF == 2) { print int($1 * 60 + $2) } else { print "0" } }')
    fi

    if [[ -z "$MAX_RSS" ]] || [[ "$MAX_RSS" == "" ]]; then
        MAX_RSS=0
    fi

    local BINARY_PATH="$PWD/build/bin/$target"
    local BINARY_SIZE_KB=0
    local IS_CUDA=false
    local HOST_FUNCTIONS=0
    local CUDA_KERNELS=0

    if [[ -f "$BINARY_PATH" ]]; then
        local CUOBJDUMP_OUTPUT
        BINARY_SIZE_KB=$(du -k "$BINARY_PATH" | cut -f1)

        CUOBJDUMP_OUTPUT=$(cuobjdump -all "$BINARY_PATH" 2>&1 || true)
        if echo "$CUOBJDUMP_OUTPUT" | grep -q "Fatbin"; then
            IS_CUDA=true
        fi

        HOST_FUNCTIONS=$(nm --defined-only "$BINARY_PATH" 2>/dev/null | wc -l || true)
        if [[ -z "$HOST_FUNCTIONS" ]] || [[ "$HOST_FUNCTIONS" == "" ]]; then
            HOST_FUNCTIONS=0
        fi

        if [[ "$IS_CUDA" == true ]]; then
            CUDA_KERNELS=$(cuobjdump -elf "$BINARY_PATH" 2>&1 | grep -c "\.nv\.info\._" || true)
            if [[ -z "$CUDA_KERNELS" ]] || [[ "$CUDA_KERNELS" == "" ]]; then
                CUDA_KERNELS=0
            fi
        fi
    fi

    echo "$target,$MAX_RSS,$ELAPSED_SEC,$BINARY_SIZE_KB,$IS_CUDA,$HOST_FUNCTIONS,$CUDA_KERNELS" > "$RESULT_FILE"

    rm -f "$TIME_OUTPUT"
    return 0
}

export -f analyze_target
export RESULTS_DIR
export PWD

if [[ $PARALLEL_JOBS -gt 1 ]]; then
    printf '%s\n' "${TARGETS_ARRAY[@]}" | \
        parallel --jobs "$PARALLEL_JOBS" --progress \
            analyze_target {} "$RESULTS_DIR"
else
    PROCESSED=0

    for target in "${TARGETS_ARRAY[@]}"; do
        PROCESSED=$((PROCESSED + 1))
        echo "[$PROCESSED/$TOTAL_TARGETS] Building: $target"

        analyze_target "$target" "$RESULTS_DIR"
    done
fi

shopt -s nullglob

FAILED=0
for result_file in "$RESULTS_DIR"/*.result; do
    if [[ -f "$result_file" ]]; then
        data=$(cat "$result_file")
        if [[ "$data" == "FAILED" ]]; then
            FAILED=$((FAILED + 1))
        else
            echo "$data" >> "$SUMMARY_FILE"
        fi
    fi
done

rm -rf "$RESULTS_DIR"

if [[ $FAILED -gt 0 ]]; then
    echo "Results saved to: $SUMMARY_FILE ($FAILED builds failed)"
    exit 1
else
    echo "Results saved to: $SUMMARY_FILE"
fi

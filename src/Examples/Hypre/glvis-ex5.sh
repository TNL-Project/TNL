#!/bin/bash

ex=ex5
keys=Aaamc

dir="${1:-.}"
mesh="$dir/$ex.mesh"
sol="$dir/$ex.sol"

if ! test -e "$mesh"; then
    echo "Error: cannot find mesh file for $ex"
    exit 1
fi

{
    echo "FiniteElementSpace"
    echo "FiniteElementCollection: H1_2D_P1"
    echo "VDim: 1"
    echo "Ordering: 0"
    echo ""
    find "$dir" -name "$ex.sol.??????" | sort | xargs cat
} > "$sol"

glvis -m "$mesh" -g "$sol" -k "$keys"

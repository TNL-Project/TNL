// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include "../../src/bitonicSort/bitonicSort.h"

#include "../benchmarker.cpp"
#include "../measure.cu"

template<typename Value>
void sorter(ArrayView<Value, Devices::Cuda> arr)
{
    bitonicSort(arr);
}
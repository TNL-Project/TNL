set(COMMON_EXAMPLES
    SumExample
    SumExampleWithFunctional
    ProductExample
    ScalarProductExample
    ScalarProductWithFunctionalExample
    MaximumNormExample
    ComparisonExample
    UpdateAndResidueExample
    MapReduceExample-1
    MapReduceExample-2
    MapReduceExample-3
    ReductionWithArgument
    ReductionWithArgumentWithFunctional
    inclusiveScanExample
    exclusiveScanExample
    inplaceInclusiveScanExample
    inplaceExclusiveScanExample
    SegmentedScanExample
)

foreach(target IN ITEMS ${COMMON_EXAMPLES})
    tnl_add_example_with_output(${target} SOURCES ${target}.cpp)
endforeach()

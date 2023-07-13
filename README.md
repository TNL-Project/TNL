[![pipeline status](https://gitlab.com/tnl-project/tnl/badges/main/pipeline.svg)](https://gitlab.com/tnl-project/tnl/commits/main)

# Template Numerical Library

TNL is a collection of building blocks that facilitate the development of
efficient numerical solvers. It is implemented in C++ using modern programming
paradigms in order to provide flexible and user friendly interface. TNL provides
native support for modern hardware architectures such as multicore CPUs, GPUs,
and distributed systems, which can be managed via a unified interface.
Visit the main [TNL web page](https://tnl-project.org/) for details.

## Components

TNL provides several optional components:

- TNL header files in the
  [src/TNL](https://gitlab.com/tnl-project/tnl/tree/main/src/TNL)
  directory.
- Various pre-processing and post-processing tools in the
  [src/Tools](https://gitlab.com/tnl-project/tnl/tree/main/src/Tools)
  directory.
- Various utilities implemented in Python in the
  [src/Python](https://gitlab.com/tnl-project/tnl/tree/main/src/Python)
  directory. Additionally, Python bindings for the C++ code are provided in
  the separate [PyTNL](https://gitlab.com/tnl-project/pytnl) repository.
- Examples of various numerical solvers in the
  [src/Examples](https://gitlab.com/tnl-project/tnl/tree/main/src/Examples)
  directory.
- Benchmarks in the
  [src/Benchmarks](https://gitlab.com/tnl-project/tnl/tree/main/src/Benchmarks)
  directory.

These components can be individually enabled or disabled and installed by a
convenient `install` script. See the [Installation][installation] section in
the documentation for details.

## Documentation

See the [full documentation][full documentation] for information about:

- [installation instructions][installation],
- [usage hints][usage],
- [Users' Guide][UsersGuide],
- [API reference manual][API],

and other documented topics.

[full documentation]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/
[installation]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/#installation
[usage]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/#usage
[UsersGuide]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/UsersGuide.html
[API]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/annotated.html

## Getting involved

The TNL project welcomes and encourages participation by everyone. While most of the work for TNL
involves programming in principle, we value and encourage contributions even from people proficient
in other, non-technical areas.

This section provides several ideas how both new and experienced TNL users can contribute to the
project. Note that this is not an exhaustive list.

- Join the __code development__. Our [GitLab issues tracker][GitLab issues] contains many ideas for
  new features, or you may bring your own. The [contributing guidelines](CONTRIBUTING.md) describe
  the standards for code contributions.
- Help with __testing and reporting problems__. Testing is an integral part of agile software
  development which refines the code development. Constructive critique is always welcome.
- Improve and extend the __documentation__. Even small changes such as improving grammar or fixing
  typos are very appreciated.
- Share __your experience__ with TNL. Have you used TNL in your own project? Please be open and
  [share your experience][contact] to help others in similar fields to get familiar with TNL. If
  you could not utilize TNL as smoothly as possible, feel free to submit a [feature request][GitLab
  issues].

Before contributing, please get accustomed with the [code of conduct][code of conduct].

[GitLab issues]: https://gitlab.com/tnl-project/tnl/-/issues
[code of conduct]: CODE_OF_CONDUCT.md
[contact]: https://tnl-project.org/#contact

## Citing

If you use TNL in your scientific projects, please cite the following papers in
your publications:

- T. Oberhuber, J. Klinkovský, R. Fučík, [TNL: Numerical library for modern parallel architectures](
  https://ojs.cvut.cz/ojs/index.php/ap/article/view/6075), Acta Polytechnica 61.SI (2021), 122-134.
- J. Klinkovský, T. Oberhuber, R. Fučík, V. Žabka, [Configurable open-source data structure for
  distributed conforming unstructured homogeneous meshes with GPU support](
  https://doi.org/10.1145/3536164), ACM Transactions on Mathematical Software, 2022, 48(3), 1-33.

## Authors

See the [list of team members](https://tnl-project.org/about/) on our website.
The [overview of contributions](https://gitlab.com/tnl-project/tnl/-/graphs/main)
can be viewed on GitLab.

## License

Template Numerical Library is provided under the terms of the [MIT License](
https://gitlab.com/tnl-project/tnl/blob/main/LICENSE).

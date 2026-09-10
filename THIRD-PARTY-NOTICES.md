# Third-party notices

This repository contains source that is not mine. It keeps its own copyright and
is not covered by this project's LICENSE.

| Path | Component | Origin |
|:--|:--|:--|
| `cg_williams_module.c` | Peter Williams' conjugate gradient method | Written by Jim Stone, Cognitive Sciences and Biology, University of Sussex, 26 June 1993. Modified by Stephen Eglen, 22 November 1995: vectors indexed from `imin` to `imax` rather than 1 to `len`, `%Lf` corrected to `%lf` in format strings, `find_2nd_deriv` signature changed, and a corrected call to `applyfunction2`. |
| `matplotlibcpp.h` | matplotlib-cpp | Benno Evers and contributors, MIT. https://github.com/lava/matplotlib-cpp |
| `filesystem.h` | ghc::filesystem | Copyright (c) 2018 Steffen Schuemann, BSD-3-Clause. https://github.com/gulrak/filesystem |

These are included because the code will not build, plot or run without them. If
you redistribute this repository, keep these notices in place.

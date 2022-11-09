# Taichi DeepPot

This is an under-developed code for interactive DeepMD visualizing and deep potential quality checking.

# Dependencies

This code depends on three main module: `taichi`, `deepmd-kit`, `ase` and `matplotlib`. The code is currently only tested on Windows 10 and Ubuntu.

Please notice that if you are using TiDP on Windows with non-development version of deepmd-kit (e.g. deepmd-kit 2.1.5), please manually adjusting `deepmd/infer/deep_eval.py` according to https://github.com/deepmodeling/deepmd-kit/pull/2054

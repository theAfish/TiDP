# TiDP: Taichi visualizer for DeepPotential

This is an under-developed code for interactive DeepMD visualizing and deep potential quality checking.

<p align="center">
  <img src="https://github.com/theAfish/TiDP/blob/main/show.png" align="center" width="500">
</p>

## Attention

This code is higly under-developed and the author has zero experience of software developing, so there are still plenties of problems:

1. Currently only 3 colors (red, green and blue) are supported for showing the atoms, and only a few atomic masses are recorded.
2. Cannot set radius for different atoms.
3. No thermostat has been implemented yet, so the MD simulation cannot be stable and usful.
4. Periodic boundary condition for non-orthogonal cells behaves unproperly.
5. float32 accuracy supported.
6. ...

Feel free to changing the code if you have ideas or suggestions on the code structure or functions ;)

## Examples

Animation example showing what can be done with TiDP:

<img src="https://github.com/theAfish/TiDP/blob/main/Animation.gif" align="center" width="1000">

## Dependencies

This code depends on four main module: [Taichi](https://github.com/taichi-dev/taichi), [DeepMD-kit](https://github.com/deepmodeling/deepmd-kit), [Atomic Simulation Environment](https://gitlab.com/ase/ase) and [matplotlib](https://github.com/matplotlib/matplotlib). The code is currently only tested on Windows 10 and Ubuntu.

**Please notice** that if you are using TiDP on Windows with non-development version of deepmd-kit python interface (e.g. deepmd-kit 2.1.5), please manually adjusting your `deepmd/infer/deep_eval.py` file according to https://github.com/deepmodeling/deepmd-kit/pull/2054

## Code Structure

The code is organized as follows:

* ``TaichiDeepPot/tidp.py``: main class of TiDP.
* ``main.py``: an example usage of TiDP

## Features
1. Running simple molecular dynamics simulation with deep potential file (TODO: adding thermostat)
2. Using probe atom to interactively check model deviation of deep potential files.

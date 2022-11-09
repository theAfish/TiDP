# TiDP: Taichi visualizer for DeepPotential

This is an under-developed code for interactive DeepMD visualizing and deep potential quality checking.

<p align="center">
  <img src="https://github.com/theAfish/TiDP/blob/main/show.png" align="center" width="500">
</p>

## Examples

Animation example showing what can be done with TiDP:

<img src="https://github.com/theAfish/TiDP/blob/main/Animation.gif" align="center" width="1000">

## Dependencies

This code depends on three main module: `taichi`, `deepmd-kit`, `ase` and `matplotlib`. The code is currently only tested on Windows 10 and Ubuntu.

**Please notice** that if you are using TiDP on Windows with non-development version of deepmd-kit python interface (e.g. deepmd-kit 2.1.5), please manually adjusting your `deepmd/infer/deep_eval.py` file according to https://github.com/deepmodeling/deepmd-kit/pull/2054

## Code Structure

The code is organized as follows:

* ``TaichiDeepPot/tidp.py``: main class of TiDP.
* ``main.py``: an example usage of TiDP

## Features
1. Running simple molecular dynamics simulation with deep potential file (TODO: adding thermostat)
2. Using probe atom to interactively check model deviation of deep potential files.

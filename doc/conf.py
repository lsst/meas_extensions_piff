"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

from documenteer.sphinxconfig.stackconf import build_package_configs
import lsst.meas.extensions.piff


_g = globals()
_g.update(build_package_configs(
    project_name='meas_extensions_piff',
    version=lsst.meas.extensions.piff.version.__version__))

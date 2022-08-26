"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

from documenteer.conf.pipelinespkg import *  # noqa: F403, import *

project = "meas_extensions_piff"
html_theme_options["logotext"] = project     # noqa: F405, unknown name
html_title = project
html_short_title = project

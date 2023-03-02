"""
Run all of the scripts necessary to replicate the figures in the paper
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

import example_CRRA
import example_log
import stat_dist
import revenue
import RA_density_check

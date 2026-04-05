#!/usr/bin/env python3
"""
Entry point for the 2D particle simulation.

Usage:
    python -m sim_2d
    python gpu/sim_2d/__main__.py
"""

from .renderer import run

if __name__ == '__main__':
    run()

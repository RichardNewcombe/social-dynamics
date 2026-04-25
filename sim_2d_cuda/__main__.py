#!/usr/bin/env python3
"""
Entry point for the 2D particle simulation (CUDA variant).

Usage:
    python -m sim_2d_cuda
"""

from .renderer import run

if __name__ == '__main__':
    run()

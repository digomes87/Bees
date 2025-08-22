#!/usr/bin/env python3
"""
Setup script for Bee Migration Analysis Project.

This script allows the project to be installed as a Python package,
making it easier to import modules and run commands.
"""

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're in the right directory
here = Path(__file__).parent.absolute()

# Read the README file
readme_file = here / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Bee Migration Analysis: Climate Impact Assessment"

# Read requirements
requirements_file = here / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "geopandas>=0.10.0",
        "folium>=0.12.0",
        "plotly>=5.0.0",
        "requests>=2.25.0",
    ]

# Read development requirements
dev_requirements_file = here / "requirements-dev.txt"
if dev_requirements_file.exists():
    with open(dev_requirements_file, "r", encoding="utf-8") as f:
        dev_requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    dev_requirements = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "isort>=5.10.0",
    ]

# Version
version = "1.0.0"

# Package metadata
setup(
    name="bee-migration-analysis",
    version=version,
    author="Diego Gomes",
    author_email="diego@example.com",
    description="Climate Impact Assessment on Bee Migration Patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diegogo/bee-migration-analysis",
    project_urls={
        "Bug Reports": "https://github.com/diegogo/bee-migration-analysis/issues",
        "Source": "https://github.com/diegogo/bee-migration-analysis",
        "Documentation": "https://github.com/diegogo/bee-migration-analysis/blob/main/README.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    keywords="bees, migration, climate, analysis, machine-learning, geospatial",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
            "coverage>=6.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "streamlit>=1.0.0",
        ],
        "api": [
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bee-analysis=bee_migration.cli:main",
            "generate-charts=bee_migration.charts:main",
            "run-analysis=bee_migration.analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bee_migration": [
            "data/*.csv",
            "data/*.json",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    test_suite="tests",
    tests_require=dev_requirements,
)

# Post-installation message
if "install" in sys.argv:
    print("\n" + "="*60)
    print("🐝 Bee Migration Analysis Project Installed Successfully! 🐝")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'bee-analysis --help' to see available commands")
    print("2. Generate charts with 'generate-charts'")
    print("3. Check the notebooks/ directory for tutorials")
    print("4. Read the documentation in README.md")
    print("\nFor development:")
    print("- Install dev dependencies: pip install -e .[dev]")
    print("- Run tests: pytest")
    print("- Format code: black src/ tests/")
    print("\nHappy analyzing! 🔬📊")
    print("="*60 + "\n")
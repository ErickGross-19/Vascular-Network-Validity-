from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vascular-network",
    version="0.1.0",
    author="Erick Gross",
    author_email="erickgross1924@gmail.com",
    description="A comprehensive package for validating, repairing, and analyzing vascular network geometries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ErickGross-19/Vascular-Network-Validity-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "trimesh>=3.9.0",
        "pymeshfix>=0.16.0",
        "pyvista>=0.32.0",
        "networkx>=2.5",
        "scipy>=1.6.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.3.0",
        "cadquery>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
        ],
    },
)

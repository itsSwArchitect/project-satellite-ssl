"""Setup configuration for satellite-ssl package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="satellite-ssl",
    version="0.1.0",
    author="PhD Researcher",
    description="Self-Supervised Learning for Satellite Imagery using SimCLR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/satellite-ssl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "umap-learn>=0.5.3",
    ],
)

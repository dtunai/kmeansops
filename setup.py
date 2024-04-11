from setuptools import setup, find_packages

setup(
    name="kmeansops",
    version="0.0.1",
    description="PyKeops Powered K-Means Clustering Algorithms Module",
    author="attophyd",
    author_email="attophyd@gmail.com",
    packages=find_packages(),
    install_requires=["torch>=0.12.0", "pykeops>=2.1.2", "matplotlib>=3.7.1"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Machine Learning",
        "Topic :: Scientific/Engineering :: Statistics",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

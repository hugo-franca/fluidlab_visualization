import setuptools

# Get version, first set unknown, then overwrite
__version__ = "0.0.1"
# exec(open('src/dropy/_version.py').read())

setuptools.setup(
        name="fluidlab_visualization",
        version=__version__,
        author="Hugo Franca",
        author_email="franca.hugo1@gmail.com",
        description="Package for processing Basilisk data using Python",
        url="https://github.com/hugo-franca/fluidlab_visualization",
        packages=setuptools.find_packages(where="src"),
        package_dir={"": "src"},
        python_requires='>=3.10',
        classifiers=[
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            ],
        install_requires=["pyfonts", "matplotlib", "numpy"],
        # setup_requires=['pytest-runner'],
        # tests_require=['pytest'],
        )

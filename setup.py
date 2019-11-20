import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsmetrics",
    version="0.0.1.4",
    author="Anderson Soares, Thales Körting",
    author_email="andersonreis.geo@gmail.com",
    description="A package to compute metrics from Satellite Image Time Series (SITS).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersonreisoares/tsmetrics/",
    packages=['tsmetrics'],
    install_requires=[
    'numpy',
    'matplotlib',
    'shapely',
    'descartes'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha"
    ],
) 
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stmetrics",
    version="0.1.4.8",
    author="Anderson Soares, Thales Körting",
    author_email="andersonreis.geo@gmail.com",
    description="A package to compute features from Satellite Image Time Series (SITS).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersonreisoares/stmetrics/",
    packages=['stmetrics'],
    install_requires=[
    'scipy',
    'sklearn',
    'pandas',
    'numpy',
    'matplotlib',
    'shapely',
    'descartes',
    'nolds',
    'dtaidistance',
    'rasterio',
    'geopandas',
    'pointpats',
    'fastremap',
    'connected-components-3d',
    'rasterstats',
    'xarray',
    'affine'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
) 
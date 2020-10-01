import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stmetrics",
    version="0.1.5",
    author="Brazil Data Cube Team",
    author_email="brazildatacube@dpi.inpe.br",
    description="A package to compute features from Satellite Image Time Series (SITS).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brazil-data-cube/stmetrics/",
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
    'pointpats==2.1.0',
    'fastremap',
    'connected-components-3d',
    'rasterstats',
    'xarray',
    'affine',
    'numba',
    'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
) 
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsmetrics",
    version="0.0.0.1",
    author="Anderson Soares",
    author_email="andersonreis.geo@gmail.com",
    description="A demo package to compute time series metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersonreisoares/tsmetrics/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha"
    ],
) 
from setuptools import setup, find_packages

setup(
    name="mvbagg",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=["tensorflow", "gpflow", "matplotlib", "scipy", "numpy", "pyjson", "contextily", "geopandas", "pandas", "sklearn", "tqdm", "shapely", "earthengine-api"]
)
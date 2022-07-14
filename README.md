# Aggregated Gaussian Processes with Multiresolution Earth Observation Covariates
Authors: Harrison Zhu*, Adam Howes, Owen van Eer, Maxime Rischard, Yingzhen Li, Dino Sejdinovic, Seth Flaxman

*Corresponding author: harrison.zhu15@imperial.ac.uk

## Paper
https://arxiv.org/pdf/2105.01460.pdf
## Data
Download scripts available in `data`
## Requirements

```python
tensorflow
gpflow
matplotlib
scipy
numpy
pyjson
contextily
geopandas
pandas
sklearn
tqdm
earthengine-api
shapely
```

Simply run `python setup.py install .`

## Experiments

### Data Processing
Obtain access to Google Earth Engine Python API first https://earthengine.google.com/new_signup/. 

Run `cd data/crops/; python download_yields.py; python processed_yields.py; cd ../../`. You may need to authenticate once.

### Model Training
Go to the respective files in `notebooks`. One caveat is that to create the inducing points (saved as a csv), you need to run `cd notebooks/usa_crops; python MVBAgg.py` first.

Results are saved in `results`.

### Visualisations
The visualisations are in the `visualisations.ipynb` notebooks for each experiment.

## License
License available at [License](LICENSE)

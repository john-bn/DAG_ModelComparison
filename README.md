# DAG_ModelComparison

This program uses Herbie & xESMF to analyze temperature & dewpoint fields from the HRRR, NAM, NBM, GFS, & other models run at NCEP, verified against either the RTMA or URMA analysis. MatPlotLib & Cartopy are then used to plot the data in a map.

The driver file is new_comparison.py. The program is run by following;

Select a model prompted by the command line output, (i.e, HRRR)
Then, a valid field to analyze. (i.e, temperature)
Then, the verification analysis source, (i.e, RTMA or URMA)
Then, a valid initiliazation hour, (i.e, 00)
Then, a valid forecast hour, (i.e, 24)

As the data is downloaded from NOMADS & AWS, no special permissions are required.
Data are downloaded automatically via Herbie and cached locally in ./data/.
For the environemnt, I recommend: conda env create -f environment.yml
This program is built for Python 3.11 (see `environment.yml`).

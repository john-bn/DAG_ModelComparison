# DAG_ModelComparison

This program uses Herbie & xESMF to analyze temperature & dewpoint fields from the HRRR, NAM, NBM, GFS, & RTMA run at NCEP. MatPlotLib & Cartopy are then used to plot the data in a map.

The driver file is new_comparison.py. The program is run by following;

Select a model prompted by the command line output, (i.e, HRRR)
Then, a valid initiliazation hour, (i.e, 00)
Then, a valid forecast hour, (i.e, 24)
Then, a valid field to analyze. (i.e, temperature)

As the data is downloaded from NOMADS & AWS, no special permissions are required.
Data are downloaded automatically via Herbie and cached locally in ./data/.
For the environemnt, I recommend: conda env create -f environment.yml
This program is built for Python version 3.13.9.

A diagram of the file tree is as follows:
DAG_ModelComparison/
├── comparator/        # Core comparison logic
├── plotting/          # Cartopy / Matplotlib figures
├── util.py/           # Shared helpers
├── scripts.py/        # Driver script

How to run:

1. Create the conda environment

conda env create -f environment.yml
conda activate new_comparator

1. Install package in editable mode

pip install -e .

1. Create your config

cp config.example.yaml config.yaml

1. Run

python scripts/run_comparator.py --config config.yaml

# astronomical-classification
Working on submission for PLAsTiCC Astronomical Classification

## Project structure  
```
.
├── data
│   ├── raw  # raw data downloaded by the script
│   └── sets  # directory for datasets
├── download-datasets.sh
├── experiments  # directory for experiments - notebooks, scripts, mlflow wrapper
├── mlruns  # directory for storing MLflow experiment logs and artifacts
├── notebooks  # directory for other notebooks
├── plasticc  # plasticc python module
├── setup.py  # plasticc and experiments modules installation instructions for pip
├── requirements.txt  # module & experiment requirements
├── requirements-dev.txt  # requirements for development
└── submissions  # folder for submission
```

## Dev setup  
1. Create virtual environment: `conda create -n plasticc python=3.6 ipykernel ipywidgets`
2. Activate the environment `source activate plasticc`
3. Install library requirements `pip install -r requirements.txt`
4. Install plasticc and experiments package in editable mode `python setup.py develop`
5. Create tmux terminal sessions for running services:
  - `source deactivate` - tmux may have issues when running from a virtual environment
  - `tmux new -s lab` -->  `source activate plasticc`; `jupyter lab`; `Ctrl+B` then `D` to exit the terminal
  - `tmux new -s mlflow` -->  `source activate plasticc`; `mlflow ui`; `Ctrl+B` then `D` to exit the terminal
6. Jupyter Lab should run on `localhost:8888` and mlflow on `localhost:5000`

## Dev workflow

### Using MLflow
See `notebooks/MLflow-Tutorial`.

### Package vs notebooks
- write new code only in the notebooks
- move code to the package when it becomes apparent that it will be reused, in particular:
  - it generates data for many experiments that are planned to be run
  - it performs the best out of available alternatives and will surely be part of a final solution
- move code to the package by creating a branch and submitting pull request to master
- notebooks can be created or pushed to the master directly, but **never push changes to other people's notebooks**

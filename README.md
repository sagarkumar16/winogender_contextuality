# winogender_contextuality

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Assessing contextuality in generative language models' resolution of gendered pronouns.

## Getting Started
### 1. Clone the Repository

```bash 
git clone https://github.com/yourusername/your-project-name.git
cd your-project-name
``` 


### 2. Set Up the Environment
In the project folder, create the ```wcenv``` environment and import all required packages using pip:

```bash
python -m venv wcvenv
source wcvenv/bin/activate
pip install -r requirements.txt
```

#### Environment Variables
Set your Huggingface Key as an environment parameter using 
```bash
export HF_KEY="your-key-here"
```

## External Data
Whether in the project folder or on a drive that can easily store up to a few gigabytes of data, create a directory 
structured as follows:

```bazaar
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
```
and update ```DATA_DIR``` in ```config.py``` to match the location of this data. 

In ```data/raw``` import the
[Winogender Templates](https://github.com/rudinger/winogender-schemas/blob/master/data/templates.tsv) and the 
[WinoPron Templates](https://github.com/uds-lsv/winopron/blob/main/data.zip). The WinoPron templates must be unzipped 
using the password provided in their README.

## Collecting Data
Once the data has been imported, you can create the data structures is primarily used for collection by running 

```bash 
python dataset.py --input-path=[path]/data/raw/templates.py --output-path=[path]/data/interim/wino_pairs.tsv
```
for the WinoGender pairs and 
```bash 
python dataset.py --input-path=[path]/data/raw/data/new_templates.py --output-path=[path]/data/interim/winopron_pairs.tsv
```
for the WinoPron pairs. 

After doing this, you can modify the bash script in ```collect_batches.py``` to run data collection via slurm or any 
other workload manager (or just remove those portions and run the batched processing script).

## Prompting
Prompts should have an assistant role such that the first token generated is the BLANK which is being resolved. 

## Data Collection
Data from models is collected by...

## Data Analysis 
Contextuality can be measured... 

## Project Organization 

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         winogender_contextuality and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── winogender_contextuality   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes winogender_contextuality a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


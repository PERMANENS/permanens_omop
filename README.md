# Permanens UPF

## Introduction
This repository contains data scripts and modeling script developed for project PERMANENS

Data scripts:
- extract data from local OMOP instances
- reformat the data and apply ATC and Conditions mappings

Modeling scripts
- build models
- create explainers
- export objects suitable for being used in the permenanes app

There will be one version of these scripts per partner. 

## Configuration
This package requires to create a Python environment. Conda should be installed in your host

```
conda env create -f environment.yml
conda activate permanens
````

## Data extraction
The following command will perform the required queries and reformating from your local OMOP instance. Note that **** should be replaced by the DB password

```
python reformat_DB -P ****
```

Requires (a version is provided in this repository)
- /mappings/mapping_conditions.tsv
- /mappings/mapping_drugs.tsv

Creates
- /matrix/predictors.yaml
- /matrix/matrix.tsv

## Model buidling
The following command will use the previously generated matrix to build a predictive model
During the process the script will informa about the model quality 

```
python model_builder_RF.py
````

Requires (generated in the previous step)
- /matrix/predictors.yaml
- /matrix/matrix.tsv

Creates
- /models/model-rf.dill

The dill file generated can be imported in the model repository of the Permanens app

## Credits

The code is created and maintained by Manuel Pastor (manuel.pastor@upf.edu) with contributions from Giacomo Ferretti

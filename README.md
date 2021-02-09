
# Quick Start

## Install the Dependencies
This implementation was tested with python 3.7.
For running the hpo experiments just doing the following is sufficient.

```
pip install -r requirements.txt
```

For the circuit experiments you have to install a simulator interface and engine:

#### NGSpice installation (for circuit simulations)
NGspice 2.7 needs to be installed separately, via this [installation link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/). Page 607 of the pdf manual on the website has instructions on how to install. Note that you might need to remove some of the flags to get it to install correctly for your machine.

#### Python Interface
Clone the following repos and add them to your `PYTHONPATH`.
```
git clone (link removed due to annonymity)
git clone (link removed due to annonymity)
export PYTHONPATH=${PYTHONPATH}:<path to blackbox_eval_engine>/src
export PYTHONPATH=${PYTHONPATH}:<path to bb_env>/src
```

### Running the experiments

The pre-trained models are saved under `saved_models`. To do the pre-training yourself
you should run the pretraining script on the corresponding data with the implemented 
hyper-parameters.

```
# ckt_sch
python scripts/pretrain_ckt.py data/ckt_d8_tpe_3000_out9D.pickle -s 10 --max_epochs 1000 -ld 32 -nu 200 -nl 3 -bs 64 --drop_out 0.5
# protein_e3 (for temporal transfer)
python scripts/pretrain_hpo.py data/protein_structure_d9_random_1000_10_cleaned.pickle -s 10 -bs 128 --lr 5e-5 --max_epochs 2000 -ld 4
# parkinsons_e3 (for temporal transfer)
python scripts/pretrain_hpo.py data/naval_propulsion_d9_random_1000_10_cleaned.pickle -s 10 -bs 128 --lr 5e-5 --max_epochs 2000 -ld 4
# naval_e3 (for temporal transfer)
python scripts/pretrain_hpo.py data/parkinsons_telemonitoring_d9_random_1000_10_cleaned.pickle -s 10 -bs 128 --lr 5e-5 --max_epochs 2000 -ld 4
# naval_e100 (for data transfer)
python scripts/pretrain_hpo.py data/naval_parkinson_d9_random_10000_10_cleaned.pickle -s 10 -bs 128 --lr 1e-4 --max_epochs 2000 -ld 4
```
### HPO

First, set `NAS_ROOT` to where [nas_benchmarks](https://github.com/automl/nas_benchmarks) is installed.
```
export NAS_ROOT=<path_to_tabular_benchmarks>
```
Then use the following commands to generate jumbo figures in the paper:

```
# protein (TT)
python scripts/run_jumbo_hpo.py -pp saved_models/protein_structure_e3  -bm protein_structure -e 100 -mi 50 -ns 5 -la 0.1
# naval (TT)
python scripts/run_jumbo_hpo.py -pp saved_models/naval_propulsion_e3  -bm naval_propulsion -e 100 -mi 50 -ns 5 -la 0.1
# parkinsons (TT)
python scripts/run_jumbo_hpo.py -pp saved_models/parkinsons_telemonitoring_e3  -bm parkinsons_telemonitoring -e 100 -mi 50 -ns 5 -la 0.1
# Naval to Parkinsons (DT)
python scripts/run_jumbo_hpo.py -pp saved_models/naval_propulsion_e100  -bm parkinsons_telemonitoring -e 100 -mi 50 -ns 5 -la 0.1
```

### Circuit Experiments

```
export NGSPICE_TMP_DIR=/tmp
python scripts/run_jumbo_ckt.py -mi 100 -ns 5 -pp saved_models/ckt_sch -la 0.05
```

## Plot the results
You can plot the results by using the following command and yaml config. Example yaml configs are 
located in `plot_configs`

```
python scripts/plot_regret.py plot_configs/plot_hpo.yaml
```

# Folder Structure

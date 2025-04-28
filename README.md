
# Replication Read Me

**Jonathan Elliott, Georges V. Houngbonon, Marc Ivaldi, and Paul Scott**

This document describes the code used to reproduce *Market Structure, Investment, and Technical Efficiencies in Mobile Telecommunications* (Elliott, Houngbonon, Ivaldi, and Scott, *Journal of Political Economy*, 2025). This document is a subset of the full replication package read me. The full replication package is available [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ADJ93R).

This document is organized as follows:

- Description of code to run model
- Description of code to clean data and produce paper results
- Example of how to run equilibrium simulation code (can skip to here), which can be adapted for other settings

Readers interested in computing an equilibrium given a specified market structure should consult section 3 and the notebook `example_counterfactual_simulation.ipynb`, which demonstrates how to compute an equilibrium.

---

## 1. Code to Run Model

The following Python files implement the model of supply and demand presented in the paper. There are three packages: `demand`, `supply`, and `welfare`, each containing several modules.

### demand/
- `demandfunctions.py`: Supplies functions that summarize demand, including market shares, elasticities, and diversion ratios
- `coefficients.py`: Supplies functions that return demand parameters for each type of consumer
- `dataexpressions.py`: Supplies functions that return expected data consumption and expected utility derived from data consumption
- `demandsystem.py`: Defines a class `DemandSystem`, containing market data and related functions
- `iteration.py`: Supplies functions used to solve for ξ’s quickly, adapted from PyBLP (Conlon and Gortmaker, 2020) to use `autograd`

### supply/
- `costs.py`: Supplies functions to back out marginal costs based on observed investment and pricing decisions
- `infrastructureequilibrium.py`: Computes the market equilibrium given firms’ costs, demand, bandwidth allocations, and spectral efficiencies
- `infrastructurefunctions.py`: Describes signal quality based on investment levels
- `priceequilibrium.py`: Describes pricing first-order conditions and elasticities
- `transmissionequilibrium.py`: Calculates average download speeds

### welfare/
- `welfare.py`: Summarizes welfare based on prices and investment levels

---

## 2. Code to Produce Paper

Running the following files in order reproduces the paper:

### Process market sizes
- `msize_alt.do`: Computes market sizes
- `agg_shares.do`: Constructs aggregate market shares

### Process choice set
- `chset.do`: Constructs the choice set

### Process download speeds
- `import_ookla.do`: Combines speed test datasets together
- `get_insee_code.py`: Determines municipality from latitude-longitude
- `ookla_quality.do`: Determines average download speeds by operator-market

### Process data consumption
- `dbar.do`: Constructs average data usage from Orange customer records

### Process incomes
- `income_alt.do`: Constructs income distributions

### Process infrastructure
- `pop_dist.R`: Computes population densities
- `base_stations.do`: Imports coordinates of base stations
- `base_stations.R`: Combines population density and base station data
- `infrastructure_calibration.do`: Constructs infrastructure data by operator-market

### Process data for demand estimation
- `demand_data.do`: Constructs dataset used for estimating demand

### Create descriptions of data
- `contracts_table.do`: Summarizes phone contract data
- `data_consumption_patterns.do`: Summarizes data consumption over the day
- `contracts_table.py`: Creates phone plans table
- `descriptive_graphs.py`: Creates descriptive graphs
- `summary_stats.py`: Creates data summary tables
- `data_consumption_patterns.py`: Data consumption statistics

### Estimate demand
- `demand.py`: Estimates demand parameters

### Process demand estimations
- `process_demand.py`: Creates tables and graphs of demand estimates

### Run counterfactual simulations
- `preprocess_counterfactuals.py`: Constructs counterfactual simulation inputs
- `counterfactuals.py`: Runs counterfactual simulations
- `process_counterfactual_arrays.py`: Constructs standard errors on counterfactual outcomes

### Process counterfactual simulations
- `process_counterfactuals.py`: Summarizes counterfactual results

### Auxiliary files
- `paths.py`: Provides file locations
- `moments.py`: Lists moments used for demand estimation
- `variancematrix.py`: Variance matrix functions
- `weightingmatrix.py`: GMM weighting matrix functions
- `estimation.py`: Functions to estimate demand via GMM
- `gmm.py`: GMM objective function

---

## 3. Equilibrium Simulation Example

This package includes a notebook `example_counterfactual_simulation.ipynb` that shows how to:

- Compute an equilibrium in prices and infrastructure given demand, products, bandwidth, costs, and market characteristics
- Modify market structure or product offerings

### Requirements to Run Notebook
- **Python version**: 3.9.19
- **NumPy**: 1.22.4
- **SciPy**: 1.13.1
- **Pandas**: 1.1.5
- **Autograd**: 1.3
- **Matplotlib**: 3.8.4

### Ways to Set Up Environment
1. Manually install the specified versions of the packages.
2. Use `environment.yml` to create a conda environment:
   ```bash
   conda env create -n telecom_test -f environment.yml
   ```
3. Use the Singularity overlay image provided in the full replication package (Linux required, details provided in the full replication package [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ADJ93R)).

---

## References

Conlon, Christopher, and Jeff Gortmaker. 2020. "Best practices for differentiated products demand estimation with pyblp." *The RAND Journal of Economics*, 51(4): 1108–1161.

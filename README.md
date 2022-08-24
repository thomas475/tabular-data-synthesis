# Tabular Data Synthesis for Mixed Data - Experiment Framework

A framework for the evaluation of different tabular data synthesis pipelines in their complete form. These include the following steps:

1. **Preprocessing**: Encoding categorical attributes and scaling numerical attributes.
2. **Data Synthesis**: Generating synthetic data with a generator trained on the preprocessed training data.
3. **_(Optional)_ Relabelling by a Black-Box Model**: Relabelling the synthetic data with a black-box model trained on the preprocessed training data. Necessary if the generator does not label the data by itself.

This synthethic data will then be added to the original training data. The augmented training data will then be used to train a white-box model, for example a shallow decision tree. The resulting performance gain is then used as a proxy for augmentation quality.

This work was done as part of my Bachelor thesis "Benchmarking Tabular Data Synthesis Pipelines for Mixed Data".

## Installation

The required packages can be installed through the 'requirements.txt' file. If you have pip installed you can simply run:

```bash
pip install -r requirements.txt
```

### PrivBayes

The PrivBayes synthesizer requires dependencies written in C++ that need to be compiled before they can be used. Make sure to have installed all the necessary dependencies to compile C++. In Linux distributions based on Ubuntu, this can be done with the following command:

```bash
sudo apt-get install build-essential
```

Navigate to the location containing the makefile for the PrivBayes compilation (`{repository_path}/framework/generators/privbayes`). Trigger the compilation from this directory with the following command:

```bash
make compile
```

This compilation results in a `privBayes.bin` binary. To use it in the framework we have to add its path to the `PRIVBAYES_BIN` environment variable:

```bash
export PRIVBAYES_BIN={repository_path}/framework/generators/privbayes/privBayes.bin
```

Always check if the `PRIVBAYES_BIN` environment variable has been set before running a long experiment with the following command:

```bash
echo $PRIVBAYES_BIN
```

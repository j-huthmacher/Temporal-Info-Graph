# Temporal-Info-Graph
This respository reflects the implementation of the Temporal Info Graph (TIG) model described in my master thesis ([https://jens-huthmacher.de/#Publications](https://jens-huthmacher.de/#Publications)) using PyTorch. This model is used for representation learning of dynamic graphs.

Besides the implementation of the TIG model the framework also contains an implementation of the STGCN from the MMSkeleton project ([https://github.com/open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton)), which can be also executed.


## Getting Started
To install and use the *Temporal Info Graph* framework follow the below described steps. However, before starting with installing the framework make sure that you have conda installed ([https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)).

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/j-huthmacher/Temporal-Info-Graph.git
   ```
2. Open the directory of the folder
   ```bash
   cd Temporal-Info-Graph
   ```
3. Create virtual environment
   ```bash
   conda create --name env_tig python=3.8.5 
   ```
4. Start virtual environment
   ```bash
    conda activate env_tig
   ```
5. Install Required Packages
   ```bash
   pip install -r requirments.txt
   ```

### Usage
The model can be used in different ways. For training the model one can use the command line interface and for evaluating or analyzing a model configuration one can use a Jupyter Notebook.

Generally you have the following options when using the CLI
```bash
Temporal Info Graph

optional arguments:
  -h, --help               show this help message and exit
  --config CONFIG          Defines which configuration should be usd. Can be a name or .json or .yml
                           file. Depending on wich execution mode is selected the config belongs to train config or evaluation config.
  --name NAME              Name of the experiment.
  --train                  Flag to select trainings mode.
  --model MODEL            [tig, stgcn]
  --eval EVAL              Option to execute evaluation mode. It requires the path to
                           the root of the experiment as argument like `--eval path/to/experiment`
  --model_name MODEL_NAME  Name of the model that is displayed in the evaluation output.
  --prep_data              Preprocess loose files from the kinetics skeleton data set.
                           Important: Set the right path to the directory in the main.py. 
                           Default output folder: ./preprocessed_data/
```

A minimal example of training the TIG model is 
```bash
python main.py --train --config single_layer
```
With this code the model is trained with the "single-layer"-configuration from `config.yml` and the results are tracked locally.

**Hint**: In case you execute the code locally, you can use the configuration for a machine with less GPU memory. I.e. instead of `single_layer` config use `single_layer_local`

**Important:** The TIG framework comes along with some data sets that are used for the action recognition experiments. When the training is started, the framework checks in the first place if the data is already located at the location specified in the configuration (default `./content/DATASET_NAME/`)

### Evaluating Trained Models
The easiest way to evalute a trained model within the TIG framework is to use the evaluation CLI.

A minimal example for evaluating a model is
```bash
python main.py --eval .\experiments\models\TIG_2_Classes_BCE_EW\50epochs\ --config 2classesSmallPortion
```

For this it is advisable to use a Jupyter Notebook (here is an example notebook).


## Feasable configuration


## Data

### Kinetics-skeleton (open-mmlab)


* https://github.com/open-mmlab/mmskeleton/blob/master/doc/SKELETON_DATA.md
* https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ

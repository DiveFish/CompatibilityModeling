# CompatibilityModeling
This is a fork of sfb833-a3/CompatibilityModeling that includes the parts that can be made publicly available. Code written by [janniss91](https://github.com/janniss91).

## Setup

For the setup of this repository simply type:

    make

This will

- set up a virtual environment for this repository,
- install all necessary project dependencies.

The lowest python version you can use is 3.7 (because the repository includes data classes).  
It is recommended to use ```Python 3.8```.

## Clean and Re-install

To reset the repository to its inital state, type:

    make dist-clean

This will remove the virtual environment and all dependencies.  
With the `make` command you can re-install them.

To remove temporary files like .pyc or .pyo files, type:

    make clean

## Packaging Info

The directory ```src/``` is a package, which is installed when you run the ```make``` command.  
Be careful to run it with the **Python module syntax** that you can see in the section ```Running``` below.

## Running

To extract all ambiguous PPs with their possible heads, type:

    python3 -m src.extract_ambiguous_pp your_input.conll your_output.tsv

**Todo**: The explanation for extract_pps must go here.

## Testing

To run the tests, type:

    python -m pytest tests/test_extract_ambiguous_pp.py

## Stored Models

3 of the different trained models have been stored:

1. Logistic Regression
2. NeuralCandidateScoringModel
3. NeuralCandidateScoringModel with Averaged nouns and lemma inputs

The models are stored in the `trained-models` directory.

The typical pytorch `state_dict` saving has been used.  
To load the model for inference, use:

```python
from src.pp_head_selection.models import NeuralCandidateScoringModel
model = NeuralCandidateScoringModel(input_dim=1076, output_dim=2)
model.load_state_dict(torch.load("trained-models/NeuralCandidateScoringModel"))

# OR:
from src.pp_head_selection.models import NeuralCandidateScoringModel
model = NeuralCandidateScoringModel(input_dim=1076, output_dim=2)
model.load_state_dict(torch.load("trained-models/NeuralCandidateScoringModel-averaged-nouns"))

# OR: 
from src.pp_head_selection.models import LogisticRegression
model = LogisticRegression(input_dim=1076, output_dim=2)
model.load_state_dict(torch.load("trained-models/LogisticRegression"))


# To inspect the model, use:
model.eval()
```

You must provide the `input_dim` and `output_dim` to the models.
The input and output dimensions are `1076` and `2` for all models.
However, note that for other feature selection processes, the input dimension might change.

You can find the training metrics and metadata for the threem models in `train-results/logs.txt`.  
For the stored models a category `Stored Model Path` is among the logged information.

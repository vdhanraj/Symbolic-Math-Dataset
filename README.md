# Symbolic Math Dataset

Flexible generator for arithmetic and symbolic-reasoning datasets, useful for training and evaluating machine-learning models on math problems.

---

## Setup

**Clone the repository**

   ```bash
   git clone https://github.com/vdhanraj/Symbolic-Math-Dataset.git
   cd Symbolic-Math-Dataset
   ``` 

**Install dependencies**

   ```bash
   conda env create -f environment.yml
   ```

or 

   ```bash
   pip install -r requirements.txt
   ```


## Generating the dataset

**Quick Start**

Generate a dataset with default settings:

```bash
python generate.py
```

**Custom Generation**
Pass command-line arguments or edit generate_dataset_config.yaml:

```bash
python generate.py \
  --possible_problems addition multiplication \
  --problem_types_training addition multiplication \
  --complexity 2 \
  --n_samples_dataset 5000 \
  --train_data_rounds 4000 \
  --val_data_rounds 500 \
  --test_data_rounds 500 \
  --string_nums True \
  --seed 42
```

If an argument is omitted, its value comes from the YAML config.

## Argument Reference

| Argument                   | Description                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| `--config`                 | Path to a YAML config file (default: `generate_dataset_config.yaml`).                           |
| `--curr_dir`               | Directory where data are written and where `generate.py` runs.                                  |
| `--possible_problems`      | Space-separated list of problem types to generate (e.g. `addition multiplication gcd lcm`).     |
| `--problem_types_training` | Subset of problem types placed in the training split (default: same as `possible_problems`).    |
| `--complexity`             | Integer difficulty level. The largest number of digits in any input/output is `complexity + 1`. |
| `--n_samples_dataset`      | Total number of problems generated **before** splitting into train/val/test.                    |
| `--train_data_rounds`      | Number of samples in the training set.                                                          |
| `--val_data_rounds`        | Number of samples in the validation set.                                                        |
| `--test_data_rounds`       | Number of samples in the test set.                                                              |
| `--string_nums`            | If `True`, numbers are spelled out in English (e.g. “one hundred and twenty-eight”).            |
| `--limit_solution_digits`  | If `True`, restricts output length for some tasks by applying `solution mod 10^(complexity+1)`. |
| `--modify_question_format` | If `True`, randomly rephrases questions to add linguistic variety.                              |
| `--x_gt_y`                 | If `True`, ensures the first operand `x` is always greater than `y`.                            |
| `--seed`                   | Random seed for reproducibility.                                                                |


## License

This project is licensed under the Apache License. See the [LICENSE](./LICENSE) file for details.




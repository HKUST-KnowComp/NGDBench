# Datagen Module of Benchmark

## Introduction

Users can use this module to generate datasets.

These datasets are used for evaluating data management systems' neural abilities in handling data's incompleteness, noise or possible bias, and the abilities to manage data by executing `UPDATE`/`ADD`/`DELETE` queries in real world scenarios.

## Directory Structure

- **gnd_datasets**: Contains ground truth data with no noise and incompleteness
- **perturbed_datasets**: Contains generated datasets with incompleteness/noise
- **perturb_record**: Records how every perturbed dataset is generated

## How to Use

### 1. Using Existing Datasets

We offer the following datasets which you can use directly.
 LDBC_SNB_BI, LDBC_FINBENCH，AltroDomain ...
 
### 2. Generating Custom Datasets

To generate your own datasets for evaluation, follow these steps:

#### Step 1: Prepare Ground Truth Data

Put the ground truth dataset under the `gnd_dataset` directory and set the `datasource` field in `configs/default_config.yaml`.

- `root_data_path` parameter: Specifies the exact directory where your ground truth data is located
- `data_file_format` parameter: Guides the program to use the correct data loader

#### Step 2: Configure Perturbation Parameters

Set the configuration parameters of the `perturbation` field in `configs/default_config.yaml`.

**Example 1: Random Incompleteness**

If you set:

```yaml
method: "random"
type: ["incompleteness"]
```

The generator will generate datasets by simulating real world data with incomplete information which is randomly and unintentionally caused, like in the process of data collecting, etc.

**Example 2: Semantic Incompleteness**

If you set:

```yaml
method: "semantic"
type: ["incompleteness"]
```

The generator will generate datasets by simulating real world data with incomplete information coming from business scenarios or is important for downstream tasks, like broken capital chain, risk mining, etc.

**Example 3: Mixed Perturbation**

You can also set mixture parameters like:

```yaml
method: "random"
type: ["incompleteness", "noise"]
```

Then the generated datasets not only simulate real data with incompleteness but also noise.

## How to Run

This project is managed by `uv`.

### Install Environment and Tools

```bash
cd ngdb_benchmark
uv sync
```

### Run the Generator

```bash
uv run data_generator.py
```
# NGDB Benchmark

Our dataset is available at [https://huggingface.co/datasets/FeifeiCS/NGDBench](https://huggingface.co/datasets/FeifeiCS/NGDBench)

## Table of Contents

- [User Guide](#user-guide)
- [Using Neo4j](#using-neo4j)
- [Status of Data Generation Modules](#status-of-data-generation-modules)
- [Usage Guide](#usage-guide)
- [Generated Datasets](#generated-datasets)

## User Guide

### 1. Convert Data to Graph Format

Convert the data into graph format (`.gpickle` or `.graphml`):

```bash
cd data_gen/graph_gen
python run.py
```

### 2. Simulate Noisy Graphs

Generate noisy graphs and record the positions of noisy nodes:

```bash
cd data_gen
python graph_generator.py
```

### 3. Build Database Containers

If you need to build your own database containers (you can also directly use the prebuilt ones):

```bash
cd pipeline/db_builder
python test_build.py
```

### 4. Generate Detection Queries

#### 4.1

On the noisy graph, generate detection queries (complex queries on noisy and clean nodes); on the clean graph, generate queries related to insert, delete, and update operations.

The queries are divided into several categories:

- **complex1**: Complex query type 1, 1/16 (three template categories: queries without aggregation, queries with aggregation, and chain queries returning a, b, d)
- **complex2**: Complex query type 2 (judgment questions), about 1w–200 queries
- **management**: Management queries (insert/delete/update), about 1w–2k queries (see the next section)

```bash
cd pipeline/query_gen
python qgen_test_noise
```

In the `query_module`, execute queries on the noisy graph (required for `complex1` and `complex2`), and then add the execution results on the noisy graph into the files of queries executed on the clean graph.

#### 4.2

(Reserved for additional query generation steps.)

### 5. Clean Query Results

Clean and post-process the query result data.

### 6. Generate NLP Descriptions

```bash
cd pipeline/handler
python translate.py
```

**Note**: Remember to modify the file name.  
For extensibility, the templates do not set explicit return limits, so `complex1` queries may all end with `return a`. Finally, you need to add an attribute to the query such as `return a._node_id` (for the LDBC dataset this is `_node_id`; for PrimeKG this can be `x_id`, `x_type`, and `x_name` returned together).

## Using Neo4j

### Basic Usage

See `pipeline/query_module/db_base.py` for details.

```python
uri = "bolt://localhost:7693"
user = "neo4j"
password = "fei123456"

# Input and output file paths
input_json_file = "../query_gen/query/ldbc_snb_finbench/noise_query_results_ldbcfin_cleaned.json"
output_json_file = "noise_execution_step1_ldbcfin_results.json"

# Create database executor
executor = DatabaseExecutor(uri, user, password)

try:
    # Connect to the database
    executor.connect()
    
    # Read queries
    queries = executor.read_queries_from_json(input_json_file)
    
    # Execute queries and compare results, enabling incremental saving
    results = executor.execute_queries_batch(
        queries, 
        compare_with_original=True,
        incremental_save=True,  # enable incremental save
        output_file_path=output_json_file
    )
finally:
    executor.close()
```

### TODO Features

- 🚧 
- 🚧 

## Status of Data Generation Modules

This section is under construction and will be updated with the progress and status of each data generation component.

## Usage Guide

For detailed instructions on how to use the `data_gen` module, please refer to [data_gen/readme.md](data_gen/readme.md).

## Generated Datasets

The currently generated datasets are stored at:

- **GPU8**: `/data/ylivm/ngdb_benchmark/data_gen/perturbed_dataset`
- **Perturbation Records**: `/data/ylivm/ngdb_benchmark/data_gen/perturb_record`


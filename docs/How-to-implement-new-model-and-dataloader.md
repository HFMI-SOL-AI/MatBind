# How to Add a New Modality

This guide demonstrates how to add a new modality for training in MatBind. This guide assumes you are working with the Materials Project as a dataset (though you can generalize this to other datasets) and are familiar with the config management tool Hydra and the deep learning framework Lightning.

---

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Step 1: Prepare Dataset](#step-1-prepare-dataset)
- [Step 2: Implement the Encoder](#step-2-implement-the-encoder)
- [Step 3: Implement the Dataset Class](#step-3-implement-the-dataset-class)
- [Step 4: Adapt Config Files](#step-4-adapt-config-files)
- [Step 5: Run and Test](#step-5-run-and-test)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Adding a new modality to MatBind involves several key steps:

1. **Dataset Preparation**: Format your data in a compatible structure
2. **Code Implementation**: Create encoder and dataset classes
3. **Configuration**: Update config files to register your new components
4. **Testing**: Validate your implementation

To maintain code organization, files should be placed in their dedicated locations following the established directory structure.

---

## File Structure

Here's where you need to place your new files or files that need to change:

```
MatBind/
├── matbind/
│   ├── data/
│   │   └── datasets/
│   │       └── your_modality/           # ← Create this folder
│   │           ├── __init__.py
│   │           └── dataset.py           # ← Your dataset class here
│   └── model/
│       └── encoders/
│           └── your_modality/           # ← Create this folder
│               ├── __init__.py
│               └── encoder.py           # ← Your encoder class here
├── configs/
│   ├── data/
│   │   └── dataset_builder/
│   │       └── default.yaml             # ← Register your dataset here (change content)
│   └── encoders/
│       └── your_modality/               # ← Create this folder
│           └── default.yaml             # ← Your encoder config here
├── tests/
│   └── test_your_modality.py            # ← Your unit tests here
```

### Naming Conventions:

- **Folder names**: Use lowercase with underscores (e.g., `elastic_tensor`)
- **File names**: Use lowercase with underscores (e.g., `dataset.py`, `encoder.py`)
- **Class names**: Use PascalCase (e.g., `ElasticTensorDataset`, `ElasticTensorEncoder`)
- **Config files**: Use lowercase with underscores (e.g., `default.yaml`)

---

## Prerequisites

Before starting, ensure you have:
- Python environment with MatBind dependencies installed
- Understanding of PyTorch Lightning modules
- Familiarity with Hydra configuration management
- Access to your modality-specific dataset

---

## Step 1: Prepare Dataset

### Data Format Requirements

Your dataset must include a `material_id` column to identify materials. The data should be saved as a single file in a table format that Polars can process.

**Supported file formats:**
- `.parquet` (recommended)
- `.csv`
- `.json`
- `.arrow`
- `.feather`

For detailed information, refer to [dataset_loading.md](dataset_loading.md).

### Example Dataset Structure

Using elastic tensor as an example:

| material_id | elastic_tensor |
|-------------|----------------|
| mp-1234     | [6x6 matrix]   |
| mp-5678     | [6x6 matrix]   |
| ...         | ...            |

###
 Naming Convention

The filename should match the column name you want to process. For the example above, name the file `elastic_tensor.parquet`.

### Data Storage

Upload your prepared dataset to:
- Dedicated project folders, or
- Hugging Face datasets hub

### Data Validation

Ensure your dataset:
- Contains no missing `material_id` values
- Has consistent data types within columns
- Includes validation labels if needed
- Is properly formatted for your specific modality

---

## Step 2: Implement the Encoder

Create your encoder class following PyTorch Lightning conventions.

### File Location
Following the [file structure](#file-structure), create your encoder at:
`matbind/model/encoders/your_modality/encoder.py`

### Basic Template

```python
import torch
import torch.nn as nn

class YourModalityEncoder(nn.Module):
    def __init__(self, embedding_dim: int, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

### Key Requirements

Your encoder must:
- Inherit from `nn.Module`
- Implement `__init__` and `forward` methods
- Output tensors with `embedding_dim` dimensions

---

## Step 3: Implement the Dataset Class

Create a PyTorch Dataset class to handle your modality data.

### File Location
Following the [file structure](#file-structure), create your dataset class at:
`matbind/data/datasets/your_modality/dataset.py`

### Basic Template

```python
import torch
from torch.utils.data import Dataset
import pandas as pd

class YourModalityDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data["your_column_name"].to_list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
```

### Key Requirements

Your dataset must:
- Inherit from `Dataset`
- Implement `__len__` and `__getitem__` methods
- Return torch tensors from `__getitem__`

---

## Step 4: Adapt Config Files

### 4.1 Register Dataset Builder

Following the [file structure](#file-structure), in `configs/data/dataset_builder/default.yaml`, add:

```yaml
your_modality_name:
  _target_: matbind.data.datasets.your_modality.YourModalityDataset
  _partial_: true
  normalize: true  # Add any default parameters
```

### 4.2 Create Encoder Configuration

Following the [file structure](#file-structure), create `configs/encoders/your_modality/default.yaml`:

```yaml
encoder:
  _target_: matbind.model.encoders.your_modality.YourModalityEncoder
  _partial_: true
  embedding_dim: 512  # Will be overridden by global embedding_size
  input_dim: 64  # Adjust based on your data
  hidden_dims: [512, 256]
  dropout: 0.1

projection_head:
  is_on: true
  freeze: false
  init_kwargs:
    dims: [512, "${encoders.embedding_size}"]  # First dim should match encoder output
    activation: LeakyReLU
    dropout: 0.1

postprocessor:
  _target_: "matbind.model.components.postprocessing.Postprocessor"
  normalize: true
  log_scaler: null
  clip_values: null  # Optional: clip extreme values
```

### 4.3 Update Experiment Configuration

In your experiment config file, add the new modality:

```yaml
data:
  modalities:
    - your_modality_name

encoders:
  your_modality_name: your_modality/default
```

---

## Step 5: Run and Test

### 5.1 Basic Testing

Test your implementation with simple checks:

```python
# tests/test_your_modality.py
test code needed
```

### 5.2 Integration Test

Run a small training test:

```bash
bash file needed
```
---

## Best Practices

### Code Organization
- Follow the established directory structure
- Use meaningful class and variable names
- Add comprehensive docstrings
- Include type hints

### Configuration Management
- Use descriptive parameter names
- Set reasonable defaults
- Document configuration options
- Group related parameters

### Performance Optimization
- Use appropriate data types (float32 vs float64)
- Implement efficient data loading
- Consider memory usage for large datasets
- Profile your code for bottlenecks

### Error Handling
- Add input validation
- Provide informative error messages
- Handle edge cases gracefully
- Log important events

---

## Troubleshooting

### Common Issues

**Data Loading Errors**
- Check file format compatibility
- Verify column names match expectations
- Ensure no corrupted data entries

**Shape Mismatches**
- Verify input dimensions match encoder expectations
- Check batch processing logic
- Validate tensor shapes at each step

**Configuration Errors**
- Ensure YAML syntax is correct
- Check that target paths exist
- Verify parameter names match class signatures

**Memory Issues**
- Reduce batch size
- Implement data streaming for large datasets
- Use efficient data types

### Debugging Tips

1. Start with a small subset of data
2. Add log statements to track tensor shapes
3. Validate data preprocessing steps independently

### Getting Help

- Check existing modality implementations for reference
- Review the project documentation
- Open an issue on the GitHub repository

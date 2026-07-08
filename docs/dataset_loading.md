# MultiModal Dataset Loading Module
## matbind.data.loading
This module provides a robust framework for loading and merging datasets, particularly suited for handling multiple modalities stored in various formats. It supports both single-file and multi-file datasets, offering flexibility and convenience in data ingestion workflows.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage and Entry Point](#usage-and-entry-point)
- [Components](#components)
  - [DATALOADING_HANDLERS](#dataloading_handlers)
  - [DatasetLoader Protocol](#datasetloader-protocol)
  - [MultiModalDatasetLoader Class](#multimodaldatasetloader-class)
  - [Utility Functions](#utility-functions)


---

## Overview

This module simplifies the process of loading datasets by:
- Supporting multiple file formats, including `.csv`, `.pickle`, `.parquet`, `.json`, and `.jsonl`.
- Allowing users to merge data from multiple modalities based on a common key (`merge_on`).
- Providing seamless integration of custom dataset loaders.

---

## Features

1. **Multi-format Support:** Handles files in several formats using predefined handlers.
2. **Directory-based Loading:** Automatically processes all supported files in a directory.
3. **Merge Capability:** Merges multiple datasets based on a common column and user-specified join type.
4. **Custom Loader Integration:** Supports user-defined dataset loading strategies.
5. **Progress Tracking:** Uses a progress bar (via `tqdm`) for user feedback during data loading.

---

## Usage and Entry Point
**`load_dataset(path_to_dataset: Path, dataset_loader: DatasetLoader | None = None) -> pd.DataFrame`**

   General-purpose dataset loader that:
   - Uses a custom `DatasetLoader` if provided.
   - Automatically selects between single-file and multi-file loaders, depending on if you provide a directory or a file path.

---

## Components

### **DATALOADING_HANDLERS**

A dictionary mapping file extensions to appropriate pandas functions for loading data:

```python
DATALOADING_HANDLERS = {
    ".csv": pd.read_csv,
    ".pickle": pd.read_pickle,
    ".pkl": pd.read_pickle,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".jsonl": lambda path: pd.read_json(path, lines=True),
}
```

This enables seamless loading of various file formats.

---

### **DatasetLoader Protocol**

A protocol that defines a contract for custom dataset loaders:

```python
class DatasetLoader(Protocol):
    def load_dataset(self, path_to_dataset: Path) -> pd.DataFrame:
        raise NotImplementedError
```

**Purpose:**
- Allows developers to define their custom dataset loading strategies that can be seamlessly integrated with the provided functions.

---

### **MultiModalDatasetLoader Class**

This class is designed to handle directory-based, multi-modality dataset loading and merging. Operates on directories structered like this (arbitrary naming):
```bash
/p/project1/solai/datasets/materials_project: tree

materials_project
├── crystal_structure.json
├── dos.json
├── pxrd128.json
├── pxrd.json
└── structure_description.json
```

#### **Constructor**

```python
def __init__(
    self,
    modalities_to_load: list[AvailableModalities] | None = None,
    merge_on: str = "material_id",
    merge_how: MergeHow = "inner",
):
```

- **`modalities_to_load`**: Specifies the modalities to load (default: `"all"`). If not provided it laods all modalities that are found inside the directory.
- **`merge_on`**: The column name to merge datasets on (default: `"material_id"`).
- **`merge_how`**: The type of merge operation (default: `"inner"`).

#### **Key Methods**

1. **`modality_should_be_loaded(self, modality: str) -> bool`**
   Checks if a given modality should be loaded based on the `modalities_to_load` parameter.

2. **`load_dataset(self, path_to_dataset: Path) -> pd.DataFrame`**
   Loads and merges datasets from a directory:
   - Iterates through files in the directory.
   - Filters unsupported formats or unwanted modalities.
   - Loads data using the appropriate handler.
   - Merges datasets into a single DataFrame.

---

### **Utility Functions**

1. **`load_dataset_from_file(path_to_dataset: Path) -> pd.DataFrame`**
   Loads a single dataset file based on its extension.

---

## Error Handling

1. **Invalid Path:** Raises a `ValueError` if the provided path is neither a file nor a directory.
2. **Unsupported Format:** Raises a `ValueError` for unsupported file formats.
3. **Empty Directory:** Raises a `ValueError` if no supported datasets are found in the directory.

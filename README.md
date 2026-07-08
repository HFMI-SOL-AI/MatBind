<div align="center">

# MatBind

</div align="center">

## 📜 Installation guide

If you want to (re)train the models, your system needs to have `CUDA` dependencies. 

Following example shows how to setup the python environment using [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv venv --python 3.11 matbind
uv pip install -e .
```

## 📁 Data availability

## 📋 Environment file

Your environment file should look like this:

```
WANDB_PROJECT="<your-wandb-project-name>"
WANDB_ENTITY="<your-wandb-account-name>"
TOKENIZERS_PARALLELISM=False
```

After you have defined your system variables in `.env`, it is read into the script as following:

```python
load_dotenv("path/to/.env")
```

## 📉 Train models

The experiment configs can be found at config
For example, to run the `train.py`

```python
python train.py 'experiment="train/example"'
```

To run the metrics on these experiments:

```python
python retrieval.py 'experiment="metrics/example"'
```

## Docs
Documentation is build with `MkDocs`:
run `mkdocs serve` to render the documentation pages

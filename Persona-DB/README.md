

## Environments
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (11.1)

## Installation
Setup `client` in `llm_api.py`.

Install [Pytorch](https://pytorch.org/) 1.9.0, then run the following in the terminal:
```shell
cd Persona-DB
conda create -n Persona-DB python=3.7 -y  # create a new conda environment
conda activate Persona-DB

chmod +x scripts/setup.sh
./scripts/setup.sh
```
Create a directories `data/OpinionQA/`. Then download content from `data.txt` and move the content there.

## Note
The running of the system might require [wandb](wandb.ai) account login.


`process_data_opinionqa2.py` is for processing (e.g., inducing) personas if you want to process for new data for OpinionQA.

`process_data_opinionqa.py` is for producing random data from OpinionQA.

## Run API
To run the system, edit the parameters with $ appended in the following and run in the terminal.
Note: --prompt_types: can choose name (without extension) from prompt/OpinionQA folder. --neighbor_topk: number of neighbors. --item_topk_collab:  number of items from collaboration, --item_topk: number of items from self. --include_rf means whether include processed persona. --num_hist: for recency. --cases is a string representing the case study including lurker_hist_100_test and longest_hist_300_test. --dataset chosen from "34", "82", or "41".

```shell
python llm_predict_opinionqa.py \
     --prompt_types $prompt_types \
     --cases $cases \
     --neighbor_topk $neighbor_topk \
     --dataset $dataset \
     --include_rf $include_rf \
     --item_topk_collab $item_topk_collab \
     --item_topk $item_topk
```

```shell
python llm_predict_opinionqa.py \
     --prompt_types extraction_refine \
     --dataset 34
```
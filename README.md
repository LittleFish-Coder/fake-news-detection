# Few-Shot Fake News Detection via Graph Neural Networks

Check [installation guide](#installation) to run the code.

## Overview
- Few-shot: N-way-K-shot 
  - N(number of classes): 2 (real/fake)
  - K(number of samples per class): 3~16
- Transductive GNN: All nodes (labeled/unlabeled train/test) are used during training for message passing.
- Loss: Calculated only on labeled nodes.
- Evaluation: Performed on test nodes.
## Metrics
- Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$

- Precision: $\frac{TP}{TP + FP}$

- Recall: $\frac{TP}{TP + FN}$

- F1-Score: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ (main metric)

## Results
![gossipcop.png](./results/gossipcop.png)
![politifact.png](./results/politifact.png)


## Dataset

[Fake_News_GossipCop](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_GossipCop)

[Fake_News_PolitiFact](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_PolitiFact)

- text(str)
- bert_embeddings(sequence): BERT embeddings of the text
- roberta_embeddings(sequence): RoBERTa embeddings of the text (768, )
- label(int): 
  - 0: real
  - 1: fake
- user_interaction(list): list of user interactions(dict)
    - content(str): content of the interaction
    - tone(str): tone of the interaction

## Usage

### GNN (HAN, HGT)
- build and train heterograph
```bash
python build_hetero_graph.py --k_shot 3 --dataset_name politifact --embedding_type deberta --edge_policy label_aware_knn --enable_dissimilar --multi_view 3
```
```bash
python train_hetero_graph.py --graph_path <graph_path> --model HGT
```

- build and train multi-graph (our work)
```bash
python build_hetero_graph_batch.py --k_shot 3 --dataset_name politifact --embedding_type deberta --edge_policy label_aware_knn --enable_dissimilar --multi_view 3
```

```bash
python train_hetero_graph_batch.py --graph_path <graph_folder> --model HGT
```

### LLM (In-context learning)
We utilize open-source LLMs to perform in-context learning.
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it)

```bash
python prompt_hf_llm.py --model_type llama --dataset_name politifact --k_shot 3
```
- dataset_name: gossipcop, politifact
- model_type: llama, gemma
- k_shot: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

### Language Model (BERT, RoBERTa)
```bash
python finetune_lm.py --model_name bert --dataset_name politifact --k_shot 3
```
- dataset_name: gossipcop, politifact
- model_name: bert, distilbert, roberta, deberta
- k_shot: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

### LSTM, MLP
```bash
python baseline_models.py --k_shot 3 --model_type MLP --dataset_name politifact
```

## Installation

- Create a new conda environment
```bash
conda create -n fakenews python=3.12
conda activate fakenews
```

- Install PyTorch (based on your CUDA version)
[(Official Doc)](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

- Install PyTorch Geometric [(Official Doc)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```bash
pip install torch-geometric
```

<!-- - Install Additional Libraries for GNN (Based on your torch version)

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
``` -->

- Install other dependencies
```bash
pip install -r requirements.txt
```
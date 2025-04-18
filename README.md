# Fake News Detection

Check [installation guide](#installation) to run the code.

## Dataset

- text: str
- embeddings: BERT embeddings of the text
- label: 
  - 0: real
  - 1: fake

### KDD2020
[Fake_News_KDD2020](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_KDD2020)

### TFG

[Fake_News_TFG](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_TFG)

### GossipCop
[Fake_News_GossipCop](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_GossipCop)

### PolitiFact
[Fake_News_PolitiFact](https://huggingface.co/datasets/LittleFish-Coder/Fake_News_PolitiFact)

## Results

### Zero Shot Classification
| Model                 | Dataset            | Accuracy     | Precision | Recall | F1     |
| --------------------- | ------------------ | ------------ | --------- | ------ | ------ |
| bart-large-mnli       | KDD2020            | 0.6232       | 0.6699    | 0.6232 | 0.5285 |
| bart-large-mnli       | GonzaloA/fake_news | 0.5461       | 0.5899    | 0.5461 | 0.4210 |
| mdeberta-v3-base-mnli | KDD2020            | 0.5972       | 0.5780    | 0.5972 | 0.5685 |
| mdeberta-v3-base-mnli | GonzaloA/fake_news | 0.6088       | 0.6164    | 0.6088 | 0.5876 |

### Fine-tuning on KDD2020

| Model                   | Accuracy           | F1                 | Loss                |
| ----------------------- | ------------------ | ------------------ | ------------------- |
| bert-base-uncased       | 0.7835671342685371 | 0.7807691946073421 | 0.46070271730422974 |
| distilbert-base-uncased | 0.7655310621242485 | 0.7673580407434882 | 0.46802181005477905 |
| roberta-base            | 0.8156312625250501 | 0.8125283885140466 | 0.4082072079181671  |

### Fine-tuning on GonzaloA/fake_news

| Model                   | Accuracy           | F1                 | Loss                 |
| ----------------------- | ------------------ | ------------------ | -------------------- |
| bert-base-uncased       | 0.9882961685351731 | 0.9882994966274701 | 0.026664618402719498 |
| distilbert-base-uncased | 0.986694591597881  | 0.9866977733534859 | 0.029809903353452682 |
| roberta-base            | 0.986694591597881  | 0.9866872535155707 | 0.024871505796909332 |

### Few Shot Learning on KDD2020

| Samples | Model                   | Accuracy           | F1                 | Loss               |
| ------- | ----------------------- | ------------------ | ------------------ | ------------------ |
| 10      | bert-base-uncased       | 0.5751503006012024 | 0.5691223701021529 | 0.6813642978668213 |
| 10      | distilbert-base-uncased | 0.5591182364729459 | 0.5447740312435575 | 0.6902174949645996 |
| 10      | roberta-base            | 0.5751503006012024 | 0.5691223701021529 | 0.6813642978668213 |
| 100     | bert-base-uncased       | 0.591182364729459  | 0.4392916815999757 | 0.6831763982772827 |
| 100     | distilbert-base-uncased | 0.591182364729459  | 0.4392916815999757 | 0.677597165107727  |
| 100     | roberta-base            | 0.591182364729459  | 0.4392916815999757 | 0.6732369661331177 |


### Few Shot Learning on GonzaloA/fake_news

| Samples | Model                   | Accuracy               | F1                 | Loss               |
| ------- | ----------------------- | ---------------------- | ------------------ | ------------------ |
| 10      | bert-base-uncased       | **0.7393125538992239** | 0.7269233177356373 | 0.570989727973938  |
| 10      | distilbert-base-uncased | 0.48367623506221513    | 0.3341564119711251 | 0.6532924771308899 |
| 10      | roberta-base            | 0.46593569052605643    | 0.2961877101553988 | 0.6716976761817932 |
| 100     | bert-base-uncased       | 0.9392632746088456     | 0.9391815954667218 | 0.3166244924068451 |
| 100     | distilbert-base-uncased | 0.9275594431440187     | 0.9273481720070647 | 0.5080302953720093 |
| 100     | roberta-base            | 0.9700628310952323     | 0.970072407552122  | 0.2685811221599579 |

### Model Comparison on Different Dataset
![GonzaloA](./src/GonzaloA.png)
![KDD2020](./src/KDD2020.png)

### Model Generalization (Cross-dataset)

We inspect the generalization of the models trained on one dataset and tested on another datase, the accuracy is used as the evaluation metric.

- Train on GonzaloA/fake_news and test on KDD2020
  | Model                   | Train Sample | In-dataset (GonzaloA/fake_news) | cross-dataset KDD2020 |
  | ----------------------- | ------------ | ------------------------------- | --------------------- |
  | bert-base-uncased       | full         | 0.9882961685351731              | 0.42685370741482964   |
  | distilbert-base-uncased | full         | 0.986694591597881               | 0.5150300601202404    |
  | roberta-base            | full         | 0.986694591597881               | 0.503006012024048     |
  | distilbert-base-uncased | 10           | 0.48367623506221513             | 0.4088176352705411    |
  | distilbert-base-uncased | 100          | 0.9275594431440187              | 0.46292585170340683   |

- Train on KDD2020 and test on GonzaloA/fake_news

  | Model                   | Train Sample | In-dataset (KDD2020) | cross-dataset GonzaloA/fake_news |
  | ----------------------- | ------------ | -------------------- | -------------------------------- |
  | bert-base-uncased       | full         | 0.7835671342685371   | 0.37058026364420354              |
  | distilbert-base-uncased | full         | 0.7655310621242485   | 0.42540347418997165              |
  | roberta-base            | full         | 0.8156312625250501   | 0.7220648022668473               |
  | distilbert-base-uncased | 10           | 0.5591182364729459   | 0.3767401749414808               |
  | distilbert-base-uncased | 100          | 0.591182364729459    | 0.5340643094739436               |


## Graph Neural Network Pipeline
1. Fetch Text-based dataser from Huggingface
2. Embed the text using BERT encoder
3. Create an empty graph - embeddings are nodes (but no edges)
4. Build edges between nodes based on the cosine similarity of embeddings
  - KNN Builder: each node is connected to its k-nearest neighbors
  - ThresholdNN Builder: each node is connected to {quantile_factor} of its neighbors
5. Train a GNN model on the graph (with only labeled_masked nodes)

### Run the GNN pipeline
1. build the embeddings graph (this should build an empty graph with only embeddings as nodes)

dataset_name currently only supports `kdd2020` 
```bash
python build_graph.py --dataset_name $dataset_name
```
if you have a pre-built graph, you can use the `--graph` argument to load the graph
```bash
python build_graph.py --prebuilt_graph graph.pt
```
choose the `edge_policy` from `knn` or `thresholdnn` (default is `thresholdnn`)
```bash
python build_graph.py --prebuilt_graph graph.pt --edge_policy thresholdnn --threshold_factor $factor
```

2. Train the GNN model
```bash
python train_graph.py --graph graph.pt
```
example:
```bash
train_graph.py --graph ./graph/kdd2020/train_full_val_full_test_full_labeled_100_knn_$k.pt
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

3. Install Additional Libraries for GNN (Based on your torch version)

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```


- Install other dependencies
```bash
pip install -r requirements.txt
```
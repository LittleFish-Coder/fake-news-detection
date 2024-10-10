import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from tqdm import tqdm
import os

def check_cuda():
    print("Check CUDA....")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print("\n\n")
    return device


def load_dataset_from_huggingface():
    # load and download the dataset from huggingface
    print("Load and download the dataset from huggingface...")
    dataset = load_dataset("LittleFish-Coder/Fake-News-Detection-Challenge-KDD-2020", download_mode="reuse_cache_if_exists", cache_dir="dataset")
    print(f"Dataset Type: {type(dataset)}")
    print(f"{dataset}")
    print(f"Dataset keys: {dataset.keys()}")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    print(f"Train dataset type: {type(train_dataset)}")
    print(f"Validation dataset type: {type(val_dataset)}")
    print(f"Test dataset type: {type(test_dataset)}")


    # First element of the train dataset
    print(f"{train_dataset[0].keys()}")
    print(f"Text: {train_dataset[0]['text']}")
    print(f"Label: {train_dataset[0]['label']}")
    print("\n\n")
    return train_dataset,val_dataset,test_dataset


def load_pretrained_LLM_and_test():
    print("Load pre-trained LLM...")
    tokenizer = AutoTokenizer.from_pretrained(f"google-bert/bert-base-uncased", clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained(f"google-bert/bert-base-uncased").to(device)
    print("Test inference LLM model...")
    # tokenize the first train dataset
    inputs = tokenizer(train_dataset[0]['text'], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    print(f"Input keys: {inputs.keys()}")
    print(f"Input ids: {inputs['input_ids']}")
    print(f"Attention mask: {inputs['attention_mask']}")


    model.config.output_hidden_states = True
    # Get model output with hidden states
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    # Now, outputs will have the hidden states
    hidden_states = outputs.hidden_states
    # The last layer's hidden state can be accessed like this
    last_hidden_state = outputs.last_hidden_state
    # take the mean of the last hidden state
    embeddings = last_hidden_state.mean(dim=1)


    # Let's check the shape of the last hidden state
    print(f"There are {len(hidden_states)} hidden states")
    print(f"Shape of the last hidden state: {last_hidden_state.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings: \n{embeddings}")
    print(type(train_dataset))
    print("\n\n")



class CustomDataset(Dataset):
    def __init__(self, texts, labels, size=None, model_name='bert-base-uncased', max_length=512):
        if size=="FULL":
            self.texts = texts
            self.labels = labels
        elif size is not None and size < len(texts):
            self.texts = texts[:size]
            self.labels = labels[:size]
        else:
            self.texts = texts
            self.labels = labels

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        # Move inputs to CUDA
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get the BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            embeddings = last_hidden_state.mean(dim=1)  # [1, 768]
            # flatten the embeddings
            embeddings = embeddings.flatten()  # [768]
            # print(f"Embeddings shape: {embeddings.shape}")

        # put the embeddings into cpu
        embeddings = embeddings.cpu()

        return embeddings, label


def get_custom_dataset(train_dataset,val_dataset,test_dataset,train_size,val_size,test_size):
    print("Get custom dataset and corresponding embedding...")
    TRAIN_SIZE = train_size
    VAL_SIZE = val_size
    TEST_SIZE = test_size

    embeddings_train_dataset = CustomDataset(texts=train_dataset['text'], labels=train_dataset['label'], size=TRAIN_SIZE)
    embeddings_val_dataset = CustomDataset(texts=val_dataset['text'], labels=val_dataset['label'], size=VAL_SIZE)
    embeddings_test_dataset = CustomDataset(texts=test_dataset['text'], labels=test_dataset['label'], size=TEST_SIZE)

    print(f"Dataset length: {len(embeddings_train_dataset)}")
    print(f"Dataset length: {len(embeddings_val_dataset)}")
    print(f"Dataset length: {len(embeddings_test_dataset)}")
    print("\n\n")

    return embeddings_train_dataset,embeddings_val_dataset,embeddings_test_dataset




class CustomGraph:
    def __init__(self, train_dataset, val_dataset, test_dataset,num_labeled):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_labeled=num_labeled
        
        self.num_nodes = len(train_dataset) + len(val_dataset) + len(test_dataset)
        self.num_features = train_dataset[0][0].shape[0]  # Assuming all embeddings have the same dimension
        
        self.graph = self._build_graph()

    def _build_graph(self):
        # Combine all embeddings
        x = torch.cat([
            torch.stack([item[0] for item in tqdm(dataset, desc=f'Processing {name} dataset')])
            for dataset, name in zip(
                (self.train_dataset, self.val_dataset, self.test_dataset), 
                ("train", "val", "test")
            )
        ])

        # Combine all labels with tqdm progress bar
        y = torch.cat([
            torch.tensor([item[1] for item in tqdm(dataset, desc=f'Processing labels for {name} dataset')])
            for dataset, name in zip(
                (self.train_dataset, self.val_dataset, self.test_dataset), 
                ("train", "val", "test")
            )
        ])

        

        # Create masks
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        train_mask[:len(self.train_dataset)] = True
        val_mask[len(self.train_dataset):len(self.train_dataset)+len(self.val_dataset)] = True
        test_mask[-len(self.test_dataset):] = True

        # random choice labeled indices
        random_indices = torch.randperm(len(self.train_dataset))[:self.num_labeled]
        labeled_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        labeled_mask[random_indices] = True


        # Create edge_index (placeholder, as we're not considering edges yet)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, 
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,labeled_mask=labeled_mask)

    def get_graph(self):
        return self.graph
    
    def __repr__(self):
        return f"CustomGraph(num_nodes={self.num_nodes}, num_features={self.num_features}, " \
               f"training_nodes={self.graph.train_mask.sum()}, " \
               f"validation_nodes={self.graph.val_mask.sum()}, " \
               f"test_nodes={self.graph.test_mask.sum()})"
    
    def __str__(self):
        return f"CustomGraph(num_nodes={self.num_nodes}, num_features={self.num_features}, " \
               f"training_nodes={self.graph.train_mask.sum()}, " \
               f"validation_nodes={self.graph.val_mask.sum()}, " \
               f"test_nodes={self.graph.test_mask.sum()})"
    
def random_choice_labeled_node(G,len_of_train_dataset,labeled_num):
    # random choice labeled indices
    random_indices = torch.randperm(len_of_train_dataset)[:labeled_num]
    labeled_mask = torch.zeros(G.num_nodes, dtype=torch.bool)
    labeled_mask[random_indices] = True
    G.labeled_mask=labeled_mask
    return G

def generate_custom_graph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset,num_labeled):
    print("Generate custom graph...")
    custom_graph = CustomGraph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset,num_labeled)
    graph_data = custom_graph.get_graph()
    print(f"Graph: {custom_graph}")
    print(f"Graph data: {graph_data}")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of features: {graph_data.num_features}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Number of training nodes: {graph_data.train_mask.sum()}")
    print(f"Number of validation nodes: {graph_data.val_mask.sum()}")
    print(f"Number of test nodes: {graph_data.test_mask.sum()}")
    print(f"Number of labeled node: {graph_data.labeled_mask.sum()}")
    print(f"len of train mask: {len(graph_data.train_mask)}")
    print(f"len of val mask: {len(graph_data.val_mask)}")
    print(f"len of test mask: {len(graph_data.test_mask)}")
    print(f"len of labeled mask: {len(graph_data.labeled_mask)}")
    print(f"len of train mask: {graph_data.train_mask}")
    print(f"len of val mask: {graph_data.val_mask}")
    print(f"len of test mask: {graph_data.test_mask}")
    print(f"len of labeled mask: {graph_data.labeled_mask}")

    print("\n\n")
    return custom_graph


def save_graph(graph_data, file_path):
    print(f"Saving graph to {file_path}...")
    torch.save(graph_data, file_path)
    print(f"Graph saved at {file_path}")

def load_graph(file_path):
    print(f"Loading graph from {file_path}...")
    graph_data = torch.load(file_path)
    print(f"Graph loaded from {file_path}")
    return graph_data


class ThresholdKNNGraphBuilder:
    def __init__(self, k=5, threshold_std_factor=1.0):
        self.k = k
        self.threshold_factor = threshold_std_factor

    def build_graph(self, x, y, train_mask, val_mask, test_mask, labeled_mask,
                    val_to_train=True, val_to_val=True,
                    test_to_test=True, test_to_train=True, val_to_test=True):
        nn = NearestNeighbors(n_neighbors=len(x), metric='cosine')
        nn.fit(x)
        distances, indices = nn.kneighbors(x)

        
        # 先計算全圖的 cosine similarity distance 統計
        flattened_distances = distances[:, 1:].flatten()  # 去除自連邊
        median_dist = np.median(flattened_distances)
        mean_dist=np.mean(flattened_distances)
        std_dist = np.std(flattened_distances)
        min_dist = np.min(flattened_distances)
        max_dist = np.max(flattened_distances)
        threshold = np.percentile(flattened_distances,self.threshold_factor)  # 計算第 n 百分位數
        
        print(f"median = {median_dist}, mean = {mean_dist}, std = {std_dist}, min = {min_dist}, max = {max_dist}, threshold = {threshold}")
        # input("Press enter to cont...")
        edge_index = []
        edge_attr = []

        out_degree=[]
        for i in tqdm(range(len(x)), desc="Building edges"):
            # 計算這個點的有效鄰居數量（小於 threshold 的鄰居數）
            valid_neighbors = [indices[i, j] for j in range(1, len(x)) if distances[i, j] < threshold]
            # print(valid_neighbors)
            if valid_neighbors:
                avg_neighbors = int(np.sqrt(len(valid_neighbors)*self.k))
                avg_neighbors = (len(valid_neighbors)+avg_neighbors)//2
            else:
                avg_neighbors = k  # 沒有小於 threshold 的鄰居就取 k
            out_degree.append(avg_neighbors)
            
            # 根據新的鄰居數量進行建邊
            for j in range(1, avg_neighbors + 1):
                neighbor = indices[i, j]
                add_edge = False

                if train_mask[i] or train_mask[neighbor]:
                    add_edge = True
                elif val_mask[i]:
                    if train_mask[neighbor] and val_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_val:
                        add_edge = True
                    elif test_mask[neighbor] and val_to_test:
                        add_edge = True
                elif test_mask[i]:
                    if train_mask[neighbor] and test_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_test:
                        add_edge = True
                    elif test_mask[neighbor] and test_to_test:
                        add_edge = True

                if add_edge:
                    edge_index.append([i, neighbor])
                    edge_attr.append(1 - distances[i, j])  # 使用相似度作為邊的屬性
        median_out_degree=np.median(out_degree)
        mean_out_degree=np.mean(out_degree)
        std_out_degree=np.std(out_degree)
        min_out_degree=np.min(out_degree)
        max_out_degree=np.max(out_degree)
        quantile_out_degree=[np.percentile(out_degree,i) for i in range(0,101,10)]
        str_stat_out_degree=f"median = {median_out_degree}, mean = {mean_out_degree}, std = {std_out_degree}, min = {min_out_degree}, max = {max_out_degree}, quantile = {quantile_out_degree}"
        # input("Press enter to cont...")
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, labeled_mask=labeled_mask),str_stat_out_degree
    
    def analyze_graph(self, graph,graph_info_path,train_size,val_size,test_size,labeled_num,k,str_stat_out_degree):
        """analyze the graph"""
        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask
        labeled_mask=graph.labeled_mask
        edge_index = graph.edge_index

        total_nodes = graph.num_nodes
        total_edges = graph.num_edges
        train_nodes = train_mask.sum().item()
        val_nodes = val_mask.sum().item()
        test_nodes = test_mask.sum().item()
        labeled_nodes=labeled_mask.sum().item()

        # analyze the edge types
        edge_types = {
            'train-train': 0, 'train-val': 0, 'train-test': 0,
            'val-val': 0, 'val-test': 0, 'test-test': 0
        }

        for edge in tqdm(edge_index.t(), desc="Analyzing edges"):
            source, target = edge
            if train_mask[source] and train_mask[target]:
                edge_types['train-train'] += 1
            elif (train_mask[source] and val_mask[target]) or (val_mask[source] and train_mask[target]):
                edge_types['train-val'] += 1
            elif (train_mask[source] and test_mask[target]) or (test_mask[source] and train_mask[target]):
                edge_types['train-test'] += 1
            elif val_mask[source] and val_mask[target]:
                edge_types['val-val'] += 1
            elif (val_mask[source] and test_mask[target]) or (test_mask[source] and val_mask[target]):
                edge_types['val-test'] += 1
            elif test_mask[source] and test_mask[target]:
                edge_types['test-test'] += 1

        with open(f"{graph_info_path}graph_info_{train_size}_{val_size}_{test_size}_k_{k}.txt",'w') as f:
            f.write(f"Total nodes: {total_nodes}\n")
            f.write(f"Total edges: {total_edges}\n")
            f.write(f"Training nodes: {train_nodes}\n")
            f.write(f"Validation nodes: {val_nodes}\n")
            f.write(f"Test nodes: {test_nodes}\n")
            f.write(f"Labeled nodes: {labeled_nodes}\n")
            f.write("\nEdge types:\n")
            for edge_type, count in edge_types.items():
                f.write(f"{edge_type}: {count}\n")
            f.write("\nStat of Out degree:\n")
            f.write(str_stat_out_degree)


class KNNGraphBuilder:
    def __init__(self, k=5):
        self.k = k

    def build_graph(self, x, y, train_mask, val_mask, test_mask,labeled_mask,
                    val_to_train=True, val_to_val=True,
                    test_to_test=True, test_to_train=True,val_to_test=True):
        nn = NearestNeighbors(n_neighbors=self.k+1, metric='cosine')
        nn.fit(x)
        distances, indices = nn.kneighbors(x)
        
        edge_index = []
        edge_attr = []
        
        for i in tqdm(range(len(x)), desc="Building edges"):
            for j in range(1,self.k+1):  # self loop
                neighbor = indices[i, j]
                
                # decide whether to add the edge
                add_edge = False
                
                if train_mask[i] or train_mask[neighbor]:
                    # train nodes can always connect to each other
                    add_edge = True
                elif val_mask[i]:
                    if train_mask[neighbor] and val_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_val:
                        add_edge = True
                    elif test_mask[neighbor] and val_to_test:
                        add_edge = True
                elif test_mask[i]:
                    if train_mask[neighbor] and test_to_train:
                        add_edge = True
                    elif val_mask[neighbor] and val_to_test:
                        add_edge = True
                    elif test_mask[neighbor] and test_to_test:
                        add_edge = True
                
                if add_edge:
                    edge_index.append([i, neighbor])
                    edge_attr.append(1 - distances[i, j])  # use similarity as the edge attribute
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,labeled_mask=labeled_mask)

    def analyze_graph(self, graph,graph_info_path,train_size,val_size,test_size,labeled_num,k):
        """analyze the graph"""
        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask
        labeled_mask=graph.labeled_mask
        edge_index = graph.edge_index

        total_nodes = graph.num_nodes
        total_edges = graph.num_edges
        train_nodes = train_mask.sum().item()
        val_nodes = val_mask.sum().item()
        test_nodes = test_mask.sum().item()
        labeled_nodes=labeled_mask.sum().item()

        # analyze the edge types
        edge_types = {
            'train-train': 0, 'train-val': 0, 'train-test': 0,
            'val-val': 0, 'val-test': 0, 'test-test': 0
        }

        for edge in tqdm(edge_index.t(), desc="Analyzing edges"):
            source, target = edge
            if train_mask[source] and train_mask[target]:
                edge_types['train-train'] += 1
            elif (train_mask[source] and val_mask[target]) or (val_mask[source] and train_mask[target]):
                edge_types['train-val'] += 1
            elif (train_mask[source] and test_mask[target]) or (test_mask[source] and train_mask[target]):
                edge_types['train-test'] += 1
            elif val_mask[source] and val_mask[target]:
                edge_types['val-val'] += 1
            elif (val_mask[source] and test_mask[target]) or (test_mask[source] and val_mask[target]):
                edge_types['val-test'] += 1
            elif test_mask[source] and test_mask[target]:
                edge_types['test-test'] += 1

        with open(f"{graph_info_path}graph_info_{train_size}_{val_size}_{test_size}_k_{k}.txt",'w') as f:
            f.write(f"Total nodes: {total_nodes}\n")
            f.write(f"Total edges: {total_edges}\n")
            f.write(f"Training nodes: {train_nodes}\n")
            f.write(f"Validation nodes: {val_nodes}\n")
            f.write(f"Test nodes: {test_nodes}\n")
            f.write(f"Labeled nodes: {labeled_nodes}\n")
            f.write("\nEdge types:\n")
            for edge_type, count in edge_types.items():
                f.write(f"{edge_type}: {count}\n")

def construct_graph_edge(graph_data,k,graph_info_path,train_size,val_size,test_size,labeled_num,EdgeConstructionPolicy):
    # Assuming x, y, train_mask, val_mask, test_mask are already defined
    builder=None
    graph=None
    if EdgeConstructionPolicy=="KNN":
        builder = KNNGraphBuilder(k)
        graph = builder.build_graph(graph_data.x, graph_data.y, graph_data.train_mask, graph_data.val_mask, graph_data.test_mask,graph_data.labeled_mask,
                                val_to_train=True, val_to_val=True,
                                test_to_test=True, test_to_train=True,val_to_test=True)
        builder.analyze_graph(graph,graph_info_path,train_size,val_size,test_size,labeled_num,k)
    elif EdgeConstructionPolicy=="ThresholdKNN":
        threshold_factor=1
        builder = ThresholdKNNGraphBuilder(k,threshold_factor)
        graph,str_stat_out_degree = builder.build_graph(graph_data.x, graph_data.y, graph_data.train_mask, graph_data.val_mask, graph_data.test_mask,graph_data.labeled_mask,
                                val_to_train=True, val_to_val=True,
                                test_to_test=True, test_to_train=True,val_to_test=True)
    

        builder.analyze_graph(graph,graph_info_path,train_size,val_size,test_size,labeled_num,k,str_stat_out_degree)
    return graph



def visualize_graph(data, show_num_nodes,train_size,val_size,test_size,labeled_num,k,plot_path):
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    for i, j in edge_index.T:
        G.add_edge(i, j)

    # Select a subset of nodes to visualize
    nodes_to_show = list(range(min(show_num_nodes, data.num_nodes)))
    subgraph = G.subgraph(nodes_to_show)

    pos = nx.spring_layout(subgraph)
    plt.figure(figsize=(12, 10))

    # Color nodes based on their mask
    node_colors = []
    for node in subgraph.nodes():
        if data.train_mask[node]:
            node_colors.append('lightblue')
        elif data.val_mask[node]:
            node_colors.append('lightgreen')
        elif data.test_mask[node]:
            node_colors.append('salmon')
        else:
            node_colors.append('gray')

    nx.draw(subgraph, pos, node_color=node_colors, node_size=50, edge_color='gray', alpha=0.7, with_labels=False)

    # Add a legend
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Train'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Validation'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='Test')],
               loc='upper right')

    plt.title(f"Graph Visualization (showing {len(subgraph)} nodes)")
    plt.savefig(f"{plot_path}Graph_Visualization_{train_size}_{val_size}_{test_size}_{labeled_num}_k_{k}_show_{len(subgraph)}_nodes.png")







def plot_tsne(x, y, train_size,val_size,test_size,labeled_num,k,plot_path,after_train=False,title="t-SNE visualization of node features"):
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    
    # Convert to numpy arrays
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_np)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_np, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    if after_train==False:
        os.makedirs(os.path.dirname(f"{plot_path}TSNE/"), exist_ok=True)
        plt.savefig(f"{plot_path}TSNE/TSNE_visualization_{train_size}_{val_size}_{test_size}_{labeled_num}.png")
    else:
        os.makedirs(os.path.dirname(f"{plot_path}TSNE/"), exist_ok=True)
        plt.savefig(f"{plot_path}TSNE/TSNE_visualization_{train_size}_{val_size}_{test_size}_{labeled_num}_k_{k}.png")







class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 256)
        self.conv2 = GCNConv(256, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x=F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x=F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.sigmoid(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 256, heads=4, concat=True)
        self.conv2 = GATConv(256 * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.tanh(x)  # Change sigmoid to tanh
        return x
    


# # Train the GCN

def get_model_criterion_optimizer(graph_data):
    model = GCN(in_channels=graph_data.num_features, hidden_channels=16, out_channels=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    return model,criterion,optimizer



def train_val_test(graph_data,model,criterion,optimizer):
    def train(data):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.labeled_mask], data.y[data.labeled_mask])
        loss.backward()
        optimizer.step()
        return loss

    def validate(data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            acc = correct / data.val_mask.sum().item()
            return acc

    def test(data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            acc = correct / data.test_mask.sum().item()
            return acc
    accuracy_record = []
    loss_record = []
    # Train the model
    best_acc=0
    for epoch in range(1, 301):
        loss = train(graph_data)
        acc = validate(graph_data)
        accuracy_record.append(acc)
        loss_record.append(loss.item())
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')
        best_acc=max(best_acc,acc)

    # Evaluate the model
    test_acc = test(graph_data)
    print(f'Test Accuracy: {test_acc:.4f}')

    return accuracy_record,loss_record,test_acc,best_acc
    


def plot_acc_loss(accuracy_record,loss_record, train_size,val_size,test_size,labeled_num,k,plot_path):
    # plot the accuracy and loss at same plot
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_record, label='Accuracy')
    plt.plot(loss_record, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.title('Accuracy/Loss')
    plt.legend()
    plt.savefig(f"{plot_path}acc_loss_{train_size}_{val_size}_{test_size}_{labeled_num}_k_{k}.png")


def plot_k_vs_result(k_range, test_result,best_acc_result,train_size,val_size,test_size,labeled_num,plot_path):
    fig, axs = plt.subplots(2, 1, figsize=(8, 18))  # 3 rows, 1 column

    # k vs test accuracy
    axs[0].plot(k_range, test_result, marker='o', linestyle='-', color='b', label="Test Accuracy")
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('Test Accuracy')
    axs[0].set_title('Test Accuracy vs. k')
    axs[0].grid(True)
    axs[0].legend()


    # k vs best accuracy
    axs[1].plot(k_range, best_acc_result, marker='o', linestyle='-', color='g', label="Best Accuracy")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('Best Accuracy')
    axs[1].set_title('Best Accuracy vs. k')
    axs[1].grid(True)
    axs[1].legend()


    plt.tight_layout()

    plt.savefig(f"{plot_path}results_vs_k_{train_size}_{val_size}_{test_size}_{labeled_num}.png")


# # Assuming you've already trained your model
def plot_tsne_after_train(graph_data,model,train_size,val_size,test_size,k):
    model.eval()
    with torch.no_grad():
        transformed_features = model(graph_data)
    plot_tsne(transformed_features, graph_data.y,train_size,val_size,test_size,k,True,"t-SNE visualization after training")

if __name__=="__main__":
    device=check_cuda()
    train_dataset,val_dataset,test_dataset=load_dataset_from_huggingface()
    load_pretrained_LLM_and_test()
    train_size=len(train_dataset)
    val_size=len(val_dataset)
    test_size=len(test_dataset)
    # labeled_num=len(train_dataset)
    labeled_num=10
    EdgeConstructionPolicy="ThresholdKNN" ##"KNN","ThresholdKNN"
    embeddings_train_dataset,embeddings_val_dataset,embeddings_test_dataset=get_custom_dataset(train_dataset,val_dataset,test_dataset,train_size,val_size,test_size)
    G=generate_custom_graph(embeddings_train_dataset,embeddings_val_dataset,embeddings_test_dataset,labeled_num)
    graph_path=f"../graph/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}/G/"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    save_graph(G.get_graph(), f"{graph_path}custom_graph_{train_size}_{val_size}_{test_size}_{labeled_num}.pt")

    # #Load graph (if needed later)
    loaded_G = load_graph(f"{graph_path}custom_graph_{train_size}_{val_size}_{test_size}_{labeled_num}.pt")
    loaded_G=random_choice_labeled_node(loaded_G,len(train_dataset),labeled_num)
    print(loaded_G)

    plot_path=f"../plot/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}/"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # plot_tsne(loaded_G.x, loaded_G.y,train_size,val_size,test_size,labeled_num,0,plot_path)


    test_result=[]
    best_acc_result=[]
    k_range=range(5,26)
    for k in tqdm(k_range, desc="Construct graph's edge..."):
        graph_info_path=f"../graph/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}_policy_{EdgeConstructionPolicy}/graph_info/"
        os.makedirs(os.path.dirname(graph_info_path), exist_ok=True)
        graph=construct_graph_edge(loaded_G,k,graph_info_path,train_size,val_size,test_size,labeled_num,EdgeConstructionPolicy)
        graph_path=f"../graph/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}_policy_{EdgeConstructionPolicy}/G/"
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        save_graph(graph,f"{graph_path}graph_{train_size}_{val_size}_{test_size}_{labeled_num}_k_{k}.pt")
        plot_path=f"../plot/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}_policy_{EdgeConstructionPolicy}/graph_visualization/"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        visualize_graph(graph,500,train_size,val_size,test_size,labeled_num,k,plot_path)

        model,criterion,optimizer=get_model_criterion_optimizer(graph)
        accuracy_record,loss_record,test_acc,best_acc=train_val_test(graph,model,criterion,optimizer)
        plot_path=f"../plot/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}_policy_{EdgeConstructionPolicy}/acc_loss/"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_acc_loss(accuracy_record,loss_record,train_size,val_size,test_size,labeled_num,k,plot_path)
        test_result.append(test_acc)
        best_acc_result.append(best_acc)
        #plot_tsne_after_train(graph,model,train_size,val_size,test_size,k)
        # input("Press enter to continue...")
    plot_path=f"../plot/train_{train_size}_val_{val_size}_test_{test_size}_labeled_{labeled_num}_policy_{EdgeConstructionPolicy}/test_result/"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_k_vs_result(k_range, test_result,best_acc_result,train_size,val_size,test_size,labeled_num,plot_path)


    
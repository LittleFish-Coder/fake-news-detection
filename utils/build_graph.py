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
import torch.nn.functional as F
from tqdm import tqdm

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
    def __init__(self, train_dataset, val_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
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

        # Create edge_index (placeholder, as we're not considering edges yet)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, 
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

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
    
def generate_custom_graph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset):
    print("Generate custom graph...")
    custom_graph = CustomGraph(embeddings_train_dataset, embeddings_val_dataset, embeddings_test_dataset)
    graph_data = custom_graph.get_graph()
    print(f"Graph: {custom_graph}")
    print(f"Graph data: {graph_data}")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of features: {graph_data.num_features}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Number of training nodes: {graph_data.train_mask.sum()}")
    print(f"Number of validation nodes: {graph_data.val_mask.sum()}")
    print(f"Number of test nodes: {graph_data.test_mask.sum()}")
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



class KNNGraphBuilder:
    def __init__(self, k=5):
        self.k = k

    def build_graph(self, x, y, train_mask, val_mask, test_mask,
                    val_to_train=True, val_to_val=True,
                    test_to_test=True, test_to_train=True):
        nn = NearestNeighbors(n_neighbors=self.k+1, metric='cosine')
        nn.fit(x)
        distances, indices = nn.kneighbors(x)
        
        edge_index = []
        edge_attr = []
        
        for i in tqdm(range(len(x)), desc="Building edges"):
            for j in range(1, self.k+1):  # skip self
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
                elif test_mask[i]:
                    if train_mask[neighbor] and test_to_train:
                        add_edge = True
                    elif test_mask[neighbor] and test_to_test:
                        add_edge = True
                
                if add_edge:
                    edge_index.append([i, neighbor])
                    edge_attr.append(1 - distances[i, j])  # use similarity as the edge attribute
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).unsqueeze(1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def analyze_graph(self, graph):
        """analyze the graph"""
        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask
        edge_index = graph.edge_index

        total_nodes = graph.num_nodes
        total_edges = graph.num_edges
        train_nodes = train_mask.sum().item()
        val_nodes = val_mask.sum().item()
        test_nodes = test_mask.sum().item()

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

        print(f"Total nodes: {total_nodes}")
        print(f"Total edges: {total_edges}")
        print(f"Training nodes: {train_nodes}")
        print(f"Validation nodes: {val_nodes}")
        print(f"Test nodes: {test_nodes}")
        print("\nEdge types:")
        for edge_type, count in edge_types.items():
            print(f"{edge_type}: {count}")

def construct_graph_edge(graph_data,k):
    # Assuming x, y, train_mask, val_mask, test_mask are already defined
    builder = KNNGraphBuilder(k)
    graph = builder.build_graph(graph_data.x, graph_data.y, graph_data.train_mask, graph_data.val_mask, graph_data.test_mask,
                                val_to_train=True, val_to_val=True,
                                test_to_test=True, test_to_train=True)

    builder.analyze_graph(graph)
    return graph



def visualize_graph(data, show_num_nodes,train_size,val_size,test_size,k):
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
    plt.savefig(f"../plot/Graph_Visualization_{train_size}_{val_size}_{test_size}_k_{k}_show_{len(subgraph)}_nodes.png")







def plot_tsne(x, y, train_size,val_size,test_size,k=0,after_train=False,title="t-SNE visualization of node features"):
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
        plt.savefig(f"../plot/TSNE/TSNE_visualization_{train_size}_{val_size}_{test_size}.png")
    else:
        plt.savefig(f"../plot/result/TSNE/TSNE_visualization_{train_size}_{val_size}_{test_size}_k_{k}.png")







class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    


# # Train the GCN

def get_model_criterion_optimizer(graph_data):
    model = GCN(in_channels=graph_data.num_features, hidden_channels=16, out_channels=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model,criterion,optimizer



def train_val_test(graph_data,model,criterion,optimizer):
    def train(data):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def test(data):
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        return acc
    accuracy_record = []
    loss_record = []
    # Train the model
    for epoch in range(1, 201):
        loss = train(graph_data)
        acc = test(graph_data)
        accuracy_record.append(acc)
        loss_record.append(loss.item())
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')

    # Evaluate the model
    test_acc = test(graph_data)
    print(f'Test Accuracy: {test_acc:.4f}')

    return accuracy_record,loss_record,test_acc
    


def plot_acc_loss(accuracy_record,loss_record, train_size,val_size,test_size,k):
    # plot the accuracy and loss at same plot
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_record, label='Accuracy')
    plt.plot(loss_record, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.title('Accuracy/Loss')
    plt.legend()
    plt.savefig(f"../plot/result/acc_loss/acc_loss_{train_size}_{val_size}_{test_size}_k_{k}.png")


def plot_k_vs_test_acc(k_range, test_result):
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, test_result, marker='o', linestyle='-', color='b', label="Test Accuracy")
    
    plt.xlabel('k')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs. k')
    
    plt.grid(True)
    plt.legend()

    plt.savefig(f"../plot/result/test_acc_vs_k_{train_size}_{val_size}_{test_size}.png")
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
    embeddings_train_dataset,embeddings_val_dataset,embeddings_test_dataset=get_custom_dataset(train_dataset,val_dataset,test_dataset,train_size,val_size,test_size)
    # G=generate_custom_graph(embeddings_train_dataset,embeddings_val_dataset,embeddings_test_dataset)
    # save_graph(G.get_graph(), f"../graph/custom_graph_{train_size}_{val_size}_{test_size}.pt")

    #Load graph (if needed later)
    loaded_G = load_graph(f"../graph/custom_graph_{train_size}_{val_size}_{test_size}.pt")
    print(loaded_G)
    plot_tsne(loaded_G.x, loaded_G.y,train_size,val_size,test_size)

    test_result=[]
    k_range=range(5,16)
    for k in tqdm(k_range, desc="Construct graph's edge..."):
        graph=construct_graph_edge(loaded_G,k)
        save_graph(graph,f"../graph/graph_{train_size}_{val_size}_{test_size}_k_{k}.pt")
        visualize_graph(graph,500,train_size,val_size,test_size,k)
        model,criterion,optimizer=get_model_criterion_optimizer(graph)
        accuracy_record,loss_record,test_acc=train_val_test(graph,model,criterion,optimizer)
        plot_acc_loss(accuracy_record,loss_record,train_size,val_size,test_size,k)
        test_result.append(test_acc)
        #plot_tsne_after_train(graph,model,train_size,val_size,test_size,k)
    plot_k_vs_test_acc(k_range, test_result)


    
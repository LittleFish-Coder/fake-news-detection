import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.gnn import GCN, GAT
from argparse import ArgumentParser
from torch.optim import Adam  # type: ignore


def show_args(args, model_name):
    print("========================================")
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print(f"\tModel name: {model_name}")
    print("========================================")

def load_graph(path: str):
    """
    Load the graph
    """
    graph = torch.load(path)
    print(f"Graph loaded from {path}")
    return graph

def get_model_criterion_optimizer(graph_data, base_model: str, dropout: bool):
    if base_model == "GCN":
        model = GCN(in_channels=graph_data.num_features, hidden_channels=16, out_channels=2, add_dropout=dropout)
    elif base_model == "GAT":
        model = GAT(in_channels=graph_data.num_features, hidden_channels=16, out_channels=2, add_dropout=dropout)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    return model, criterion, optimizer

def train_val_test(graph_data, model, criterion, optimizer, n_epochs=300, output_dir='weights', model_name='model'):

    os.makedirs(output_dir, exist_ok=True)

    def train(graph_data):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out[graph_data.labeled_mask], graph_data.y[graph_data.labeled_mask])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        # in training phase, we only learn from the labeled_mask data
        pred = out[graph_data.labeled_mask].argmax(dim=1)  # Predicted labels
        correct = (pred == graph_data.y[graph_data.labeled_mask]).sum().item()  # Correct predictions
        train_acc = correct / graph_data.labeled_mask.sum().item()  # Training accuracy
    
        return train_acc, loss.item()

    def validate(graph_data):
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index)
            val_loss = criterion(out[graph_data.test_mask], graph_data.y[graph_data.test_mask])
            val_pred = out[graph_data.test_mask].argmax(dim=1)
            val_correct = (val_pred == graph_data.y[graph_data.test_mask]).sum().item()
            val_acc = val_correct / graph_data.test_mask.sum().item()
        return val_acc, val_loss.item()

    def test(graph_data):
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index)
            test_pred = out[graph_data.test_mask].argmax(dim=1)
            test_correct = (test_pred == graph_data.y[graph_data.test_mask]).sum().item()
            test_acc = test_correct / graph_data.test_mask.sum().item()
        return test_acc
        

    # record
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    best_loss = float('inf')

    # Train the model
    for epoch in range(n_epochs):

        train_acc, train_loss = train(graph_data)
        val_acc, val_loss = validate(graph_data)

        # Store metrics for each epoch
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{output_dir}/{model_name}.pt")
            print(f"Model saved at {output_dir}/{model_name}.pt")

    # Evaluate the model
    test_acc = test(graph_data)
    print(f'Test Accuracy (with last epoch model): {test_acc:.4f}')

    return train_accs, train_losses, val_accs, val_losses, test_acc

def test(model, graph_data):
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        test_pred = out[graph_data.test_mask].argmax(dim=1)
        test_correct = (test_pred == graph_data.y[graph_data.test_mask]).sum().item()
        test_acc = test_correct / graph_data.test_mask.sum().item()
    return test_acc

def save_results(train_accs, train_losses, val_accs, val_losses, last_test_acc, inference_test_acc, model_name, output_dir):
    with open(f"{output_dir}/{model_name}_results.txt", "w") as f:
        f.write(f"Train Accuracy: {train_accs}\n")
        f.write(f"Train Loss: {train_losses}\n")
        f.write(f"Validation Accuracy: {val_accs}\n")
        f.write(f"Validation Loss: {val_losses}\n")
        f.write(f"Test Accuracy (with last epoch model): {last_test_acc}\n")
        f.write(f"Test Accuracy (with best model): {inference_test_acc}\n")

def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name, output_dir):
    # plot the accuracy and loss in two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot accuracy
    ax1.plot(train_accs, label='Train Accuracy')
    ax1.plot(val_accs, label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()

    # Plot loss
    ax2.plot(train_losses, label='Train Loss')
    ax2.plot(val_losses, label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    ax2.legend()
    plt.title(f"Accuracy and Loss of {model_name.replace('_', ' ')}")
    plt.savefig(f"{output_dir}/{model_name}_acc_loss.png")

def plot_tsne(graph_data, model, model_name):

    # create tsne dir
    os.makedirs("tsne", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # before inference
    transformed_features = graph_data.x.cpu().numpy()
    transformed_features = TSNE(n_components=2).fit_transform(transformed_features)
    axes[0].scatter(transformed_features[:, 0], transformed_features[:, 1], c=graph_data.y.cpu(), cmap='coolwarm', s=3)
    axes[0].set_title("Before Training")

    # after inference
    model.eval()
    with torch.no_grad():
        transformed_features = model(graph_data.x, graph_data.edge_index)
        transformed_features = transformed_features.cpu().numpy()
        transformed_features = TSNE(n_components=2).fit_transform(transformed_features)
    axes[1].scatter(transformed_features[:, 0], transformed_features[:, 1], c=graph_data.y.cpu(), cmap='coolwarm', s=3)
    axes[1].set_title("After Training")

    fig.suptitle(f"t-SNE of {model_name.replace('_', ' ')}")
    plt.savefig(f"tsne/{model_name}.png")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train graph by given graph data")
    parser.add_argument("--graph", type=str, help="path to graph data", required=True)
    parser.add_argument("--base_model", type=str, default="GCN", help="base model to use", choices=["GCN", "GAT"])
    parser.add_argument("--dropout", action="store_true", help="Enable dropout")
    parser.add_argument("--no-dropout", dest="dropout", action="store_false", help="Disable dropout")
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--train_all", action="store_true", help="train all data (ignore labeled_mask)")
    parser.add_argument("--output_dir", type=str, default="weights", help="path to save the weights")

    args = parser.parse_args()
    graph = args.graph
    base_model = args.base_model
    dropout = args.dropout
    n_epochs = args.n_epochs
    train_all = args.train_all
    output_dir = args.output_dir

    # show arguments
    model_name = f"{base_model}_{graph.split('/')[-1].split('.')[0]}"
    if train_all:
        model_name = f"{model_name}_train_all"
    show_args(args, model_name)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # load graph
    graph = load_graph(graph)

    if train_all:
        print("Training all training nodes (ignoring labeled_mask)")
        graph.labeled_mask = graph.train_mask

    # get model, criterion, optimizer
    model, criterion, optimizer = get_model_criterion_optimizer(graph, base_model, dropout)

    # train, validate, test
    train_accs, train_losses, val_accs, val_losses, last_test_acc = train_val_test(graph, model, criterion, optimizer, n_epochs, output_dir, model_name)

    # load the best model
    model.load_state_dict(torch.load(f"{output_dir}/{model_name}.pt", weights_only=True))

    # test the best model
    inference_test_acc = test(model, graph)
    print(f"Test accuracy (with best model): {inference_test_acc}")

    # save results
    save_results(train_accs, train_losses, val_accs, val_losses, last_test_acc, inference_test_acc, model_name, output_dir)

    # plot the accuracy and loss at same plot
    plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name, output_dir)
    
    # plot t-SNE, before and after training
    plot_tsne(graph, model, model_name)
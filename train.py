import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

"""
How to run this script:

Basic (single process):
    python3 train.py --epochs 10 --batch_size 64

Distributed (multi-CPU/node, e.g. on Oracle Cloud):
    torchrun --nproc_per_node=NUM_PROCESSES train.py --epochs 10 --batch_size 64

Arguments:
    --data_dir      Path to the data directory (default: 'data')
    --epochs        Number of training epochs (default: 10)
    --batch_size    Batch size for training (default: 64)
    --lr            Learning rate (default: 0.001)
    --local_rank    (Set automatically by torchrun for distributed training)
    --world_size    (Set automatically by torchrun for distributed training)

Example for 4 CPUs:
    torchrun --nproc_per_node=4 train.py --epochs 20 --batch_size 128

"""



def load_data(data_dir, columns=None):

    data = pd.read_csv(os.path.join(data_dir, 'UNSW_NB15_training-set.csv'))
    if columns is None:
        columns = data.columns

    acceptable = ['1.csv', '2.csv', '3.csv', '4.csv']
    for filename in os.listdir(data_dir):
        if any(filename.endswith(suffix) for suffix in acceptable):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path, header=0, names=columns, skiprows=1, low_memory=False)
            data = pd.concat([data, df], ignore_index=True)
    return data

def preprocess_data(data):
    data = data.drop(['attack_cat', 'service', 'id'], axis=1, errors='ignore')

    data = data.dropna(axis=0, thresh=int(data.shape[1] * 0.80))

    labels = data['label'].values.astype(np.float32)
    features = data.drop(columns=['label'], axis=1)

    categorical_columns = ['proto', 'state']
    encoder = OneHotEncoder(sparse=False)
    cat_data = encoder.fit_transform(features[categorical_columns])

    num_data = features.drop(columns=categorical_columns).values.astype(np.float32)

    from scipy import sparse
    X = sparse.hstack([num_data, cat_data]).tocsr()
    return X, labels, encoder

def create_dataloaders(X, y, batch_size, is_distributed, rank=0, world_size=1):

    X = torch.tensor(X.todense(), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    return loader, sampler

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, loader, criterion, optimizer, device, epoch, sampler=None):
    model.train()
    running_loss = 0.0
    if sampler is not None:
        sampler.set_epoch(epoch)
    for batch_X, batch_Y in loader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        outputs = model(batch_X).view(-1)
        loss = criterion(outputs, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}")
    return epoch_loss

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    TP = FP = FN = 0
    with torch.no_grad():
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().view(-1)
            labels = batch_Y.view(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            TP += ((preds == 1) & (labels == 1)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()
    accuracy = correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1_score*100:.2f}%")
    return accuracy, precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    is_distributed = 'RANK' in os.environ or args.world_size > 1
    rank = int(os.environ.get('RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))

    if is_distributed:
        torch.distributed.init_process_group(backend='gloo')
    device = torch.device('cpu')

    # Only rank 0 prints
    def log(*a, **kw):
        if rank == 0:
            print(*a, **kw)

    log('Loading data...')
    data = load_data(args.data_dir)
    log('Preprocessing data...')
    X, y, encoder = preprocess_data(data)
    log(f'Feature shape: {X.shape}, Labels: {y.shape}')

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Standardize numerical features (only on dense part)
    scaler = StandardScaler()
    X_train_dense = scaler.fit_transform(X_train[:, :X.shape[1] - encoder.categories_[0].size - encoder.categories_[1].size])
    X_test_dense = scaler.transform(X_test[:, :X.shape[1] - encoder.categories_[0].size - encoder.categories_[1].size])
    # Recombine with sparse one-hot
    from scipy import sparse
    X_train = sparse.hstack([X_train_dense, X_train[:, X.shape[1] - encoder.categories_[0].size - encoder.categories_[1].size:]]).tocsr()
    X_test = sparse.hstack([X_test_dense, X_test[:, X.shape[1] - encoder.categories_[0].size - encoder.categories_[1].size:]]).tocsr()

    train_loader, train_sampler = create_dataloaders(X_train, y_train, args.batch_size, is_distributed, rank, world_size)
    test_loader, _ = create_dataloaders(X_test, y_test, args.batch_size, False)

    input_dim = X_train.shape[1]
    model = Net(input_dim).to(device)
    if is_distributed:
        model = DDP(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, train_sampler)
        if rank == 0:
            evaluate(model, test_loader, device)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')
        log('Model saved to model.pth')

if __name__ == '__main__':
    main()

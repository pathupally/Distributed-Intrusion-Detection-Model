import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_data():
    # Load training dataset with headers
    data = pd.read_csv('data/UNSW_NB15_training-set.csv') # Load dataset
    
    # Get column names from training data
    column_names = data.columns.tolist()
    
    # List of additional CSV files to load
    additional_files = [
        'data/UNSW-NB15_1.csv',
        'data/UNSW-NB15_2.csv', 
        'data/UNSW-NB15_3.csv',
        'data/UNSW-NB15_4.csv',
    ]
    
    # Load and append additional CSV files
    for file_path in additional_files:
        try:
            additional_data = pd.read_csv(file_path, header=None, names=column_names)
            data = pd.concat([data, additional_data], ignore_index=True)
            print(f"Loaded {file_path} - Total rows: {len(data)}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, skipping...")
    
    
    # Handle missing values
    data = data.fillna('unknown')
    
    # Find all non-numeric columns (except label and attack_cat)
    labels = data['label'].values # Extract labels first
    features = data.drop(columns=['label', 'attack_cat'], axis=1) # Extract features
    
    # Identify categorical columns automatically
    categorical_columns = []
    for col in features.columns:
        if features[col].dtype == 'object' or features[col].dtype == 'string':
            categorical_columns.append(col)
    
    print(f"Found categorical columns: {categorical_columns}")
    
    # Use LabelEncoder for all categorical columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Convert all remaining columns to numeric, coercing errors to NaN
    for col in features.columns:
        if col not in categorical_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
    
    # Fill any remaining NaN values with 0
    features = features.fillna(0)
    
    print(f"Final feature shape: {features.shape}")
    print(f"Feature data types: {features.dtypes.value_counts()}")
    
    return features, labels

def pre_processing(features, labels):
    ''' Training and Testing Data '''
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels) # Split data into training and testing sets
    scaler = StandardScaler() # Initialize scaler
    X_train = scaler.fit_transform(X_train) # Fit and transform training data
    X_test = scaler.transform(X_test) # Transform testing data
    
    return X_train, X_test, Y_train, Y_test

def data_loader(X_train, X_test, Y_train, Y_test):
    # 6. Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(Y_test, dtype=torch.float32)

    # 7. Create TensorDataset for convenient data handling
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    # 8. Use DataLoader to batch and shuffle the data
    batch_size = 64  # industry-standard batch sizes are powers of 2 (32, 64, 128, etc.)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

# 9. Define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        # Define layers:
        self.fc1 = nn.Linear(input_size, 64)   # fully connected layer 1: input -> 64 hidden units
        self.fc2 = nn.Linear(64, 1)           # fully connected layer 2: 64 -> 1 output
    
    def forward(self, x):
        # Define the forward pass (how data moves through the network)
        x = F.relu(self.fc1(x))   # apply ReLU activation after first layer
        x = self.fc2(x)           # second layer (output logits)
        return x

def train_model(train_loader, train_dataset, input_dim):
    # 10. Initialize the model
    model = Net(input_dim)
    print(model)

    device = torch.device('cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            outputs = model(batch_X)
            outputs = outputs.view(-1)
            
            loss = criterion(outputs, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return model

def evaluate_model(model, test_loader):
    device = torch.device('cpu')
    model.eval()

    correct = 0
    total = 0
    TP = FP = FN = 0

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            outputs = model(batch_X)
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            # Flatten predictions and labels to 1D
            preds = preds.view(-1)
            labels = batch_Y.view(-1)
            # Count correct predictions
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            
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

def main():
    # Load and preprocess data
    features, labels = load_data()
    X_train, X_test, Y_train, Y_test = pre_processing(features, labels)
    
    # Create data loaders
    train_loader, test_loader, train_dataset, test_dataset = data_loader(X_train, X_test, Y_train, Y_test)
    
    # Get input dimension
    input_dim = X_train.shape[1]
    
    # Train the model
    model = train_model(train_loader, train_dataset, input_dim)
    
    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
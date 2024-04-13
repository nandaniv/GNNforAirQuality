#Observation made is train and test RMSE is very close
import pandas as pd
import numpy as np
import torch
import os
import tensorflow as tf
import random 
from tensorflow import keras
from keras.layers import LSTM, GRU, Dense 
from keras.models import Sequential
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN2
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch, StaticGraphTemporalSignal
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats

# Creating adjacency matrix
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_adj = pd.read_csv('/content/project-master/data/distanceNodes.txt', encoding='utf8', delimiter='\t')

# create new column called Distance_KM, which calculates distance between nodes in km
data_adj['Distance_KM'] = data_adj['NEAR_DIST'] / 1000

# Change the nodes id
replacement_mapping_dict = {
    0: 141, 1: 143, 2: 195, 3: 181, 4: 5, 5: 88, 6: 138, 7: 254, 8: 142, 9: 113, 10: 192, 11: 243, 12: 78, 13: 92, 
    14: 177, 15: 126, 16: 228, 17: 47, 18: 220, 19: 72, 20: 281, 21: 323, 22: 217, 23: 296
}
data_fin = data_adj[["IN_FID", "NEAR_FID"]].replace(replacement_mapping_dict)

# Replace columns with the new created columns
data_adj['IN_FID'] = data_fin["IN_FID"]
data_adj['NEAR_FID'] = data_fin["NEAR_FID"]

# create Adjacency matrix and replace column and index names according to grid cells id where air quality stations are located
am = pd.DataFrame(np.zeros(shape=(24, 24)))
am.rename(columns=replacement_mapping_dict, index=replacement_mapping_dict, inplace=True)
adj_mat = am.sort_index(axis=1)
adj_mat_complete = adj_mat.sort_index()

# Adjacency matrix
for i in data_adj.IN_FID.unique():
    for j in data_adj.NEAR_FID.unique():
        if i == j:
            adj_mat_complete.at[i, j] = 0
        else:      
            adj_mat_complete.at[i, j] = 1 / data_adj.loc[(data_adj['IN_FID'] == i) & (data_adj['NEAR_FID'] == j)]['Distance_KM']

print(adj_mat_complete)

# Set the seed for reproducibility
rnd_seed = 11

def set_seed(seed_num) -> None:
    random.seed(seed_num)
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    tf.experimental.numpy.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed_num)
    print(f"Random seed set as {seed_num}")

# Read the file containing nodes (air quality monitoring stations) features as pandas dataframe
Mad_data_2019 = pd.read_csv('/content/project-master/data/Mad_Station_2019.csv')
Mad_data_2022 = pd.read_csv('/content/project-master/data/Mad_Station_2022.csv')

# Delete 'windDir' column as we will not need it for further analysis.
Mad_data_2019 = Mad_data_2019.drop(['windDir'], axis=1)
Mad_data_2022 = Mad_data_2022.drop(['windDir'], axis=1)

# Convert dataframes to numpy arrays
fin_data = Mad_data_2019.to_numpy().reshape(-1, 24, 18)
fin_data_test = Mad_data_2022.to_numpy().reshape(-1, 24, 18)
fin_data = fin_data[:100, :, :]
fin_data_test = fin_data_test[:100, :, :]
fin_data_test.shape

# Function to return data in original scale (reversing Z score)
def reverse_zscore(pandas_series, mean, std):
    '''Mean and standard deviation should be of original variable before standardisation'''
    yis = pandas_series * std + mean
    return yis

# Convert adjacency dataframe to numpy
adj = adj_mat_complete.to_numpy()
# standardise train data

data = fin_data.transpose(
            (1, 2, 0)
        )
data = data.astype(np.float32)

# standardise (via Z-Score Method)
means = np.mean(data, axis=(0, 2))
data_norm= data-means.reshape(1, -1, 1)
stds = np.std(data_norm, axis=(0, 2))
data_norm= data_norm/ stds.reshape(1, -1, 1)

#to convert adjacency matrix and standardised train data to torch
adj = torch.from_numpy(adj)
data_norm= torch.from_numpy(data_norm)


# standardise test data using means and standard deviation of train set

data_test = fin_data_test.transpose(
            (1, 2, 0)
        )
data_test = data_test.astype(np.float32)
data_test_norm= data_test- means.reshape(1, -1, 1)
data_test_norm= data_test_norm/ stds.reshape(1, -1, 1)

#to convert standardised test data to torch
data_test_norm = torch.from_numpy(data_test_norm)

adj.shape

data_test_norm.shape

#from adjacency matrix extract the edge indices and edge weights

edge_indices, values = dense_to_sparse(adj)
edge_indices = edge_indices.numpy()
values = values.numpy()
edges = edge_indices
edge_weights = values
batch =64

edges.shape

edge_weights.shape

class MadridDatasetLoader(object):
    """The dataset is based on 24 stations (nodes) each having 18 features (nodal features) 
    and 276 edges connecting each pair of nodes, the edge weights are the distance between the edges.
    """

    def __init__(self, data_norm, edges, edge_weights, batch):
        super(MadridDatasetLoader, self).__init__()
        
        self.data_norm = data_norm
        self.edges = edges 
        self.edge_weights= edge_weights
        self.batch = batch


    def _generate_task(self, num_timesteps_in: int = 6, num_timesteps_out: int = 6):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        time_steps_starter =   0 # it can be assigned as one of the following {0, 12, 24, 36}
        indices = [
            (i, i +time_steps_starter+ (num_timesteps_in + num_timesteps_out))
            for i in range(self.data_norm.shape[2] - (time_steps_starter+num_timesteps_in + num_timesteps_out) + 1)
        ]
        print(indices)
        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.data_norm[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.data_norm[  :, 0, i + num_timesteps_in +time_steps_starter: j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 6, num_timesteps_out: int = 6
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for the dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The forecasting dataset.
        """
        
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignalBatch(
            self.edges, self.edge_weights, self.features, self.targets, self.batch
        )

        return dataset

loader = MadridDatasetLoader(data_test_norm, edges, edge_weights, batch)
dataset_test = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
print("Dataset type:  ", dataset_test)


loader = MadridDatasetLoader(data_norm, edges, edge_weights, batch)
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
print("Dataset type:  ", dataset)


# Standardise train data
data = fin_data.transpose((1, 2, 0))
data = data.astype(np.float32)
unit_num  = 256  # This is number of units in order to construct the architecture of the A3T-GCN. 
import torch
import torch.nn.functional as F

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, hidden_units):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=unit_num, 
                           periods=periods)
        # Hidden layers
        self.hidden_layer1 = torch.nn.Linear(unit_num, hidden_units)
        self.hidden_layer2 = torch.nn.Linear(hidden_units, hidden_units)
        self.hidden_layer3 = torch.nn.Linear(hidden_units, hidden_units)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(hidden_units, periods)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.hidden_layer1(h)  # Apply hidden layer 1
        h = F.relu(h)
        h = self.hidden_layer2(h)  # Apply hidden layer 2
        h = F.relu(h)
        h = self.hidden_layer3(h)  # Apply hidden layer 3
        h = F.relu(h)
        h = self.linear(h)
        return h

# Create an instance of TemporalGNN with 18 node features, 12 periods, and 64 hidden units
model = TemporalGNN(node_features=18, periods=12, hidden_units=64)

# Instantiate model

#model = TemporalGNN(node_features=18, periods=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
model.train()

print("Running training...")
for epoch in range(1): 
  loss = 0
  step = 0
  for snapshot in dataset:
    snapshot = snapshot.to(device)
        # Get model predictions
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # Mean Squared Error
    yhat_reverse = reverse_zscore(y_hat, means[0], stds[0])
    snapshot.y_reverse = reverse_zscore(snapshot.y, means[0], stds[0])
    loss = loss + torch.mean((y_hat-snapshot.y)**2) 
   # loss = loss + torch.mean((y_hat-snapshot.y)**2) 
    loss = loss + torch.sqrt(torch.mean((yhat_reverse-snapshot.y_reverse)**2)) 

    step += 1
        
loss = loss / (step+1)
loss = loss.item()
print("Train RMSE: {:.4f}".format(loss))


# Define the input, hidden, and output channel numbers
input_channels = 18  # Assuming 18-dimensional node features
hidden_channels = 64  # Number of hidden channels in the graph convolutional layers
output_channels = 12  # Number of output channels in the linear layer

# Define other parameters
periods = 18  # Number of periods

# Instantiate the TemporalGNN class
model = TemporalGNN(node_features=18, periods=12, hidden_units=64)

#testing
#model = TemporalGNN(node_features=18, periods=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.eval()
loss = 0
step = 0

# Store for analysis
predictions = []
labels = []





print("Running testing...")
for epoch in range(1): 
  loss = 0
  step = 0
  for snapshot in dataset_test:
    snapshot = snapshot.to(device)
        # Get model predictions
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        # Mean Squared Error

    yhat_reverse = reverse_zscore(y_hat, means[0], stds[0])
    snapshot.y_reverse = reverse_zscore(snapshot.y, means[0], stds[0])
    loss = loss + torch.mean((y_hat-snapshot.y)**2) 
   # loss = loss + torch.mean((y_hat-snapshot.y)**2) 
    loss = loss + torch.sqrt(torch.mean((yhat_reverse-snapshot.y_reverse)**2))  
    step += 1
        
loss = loss / (step+1)
loss = loss.item()
print("Test RMSE: {:.4f}".format(loss))

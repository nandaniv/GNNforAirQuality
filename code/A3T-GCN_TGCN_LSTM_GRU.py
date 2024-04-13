#install pytorch and all related libraries
#it will take some time

# !pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
# !pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
# !pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
# !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
# !pip install torch-geometric
# !pip install torch-geometric-temporal


# import torch
# CUDA_VISIBLE_DEVICES=0,1


# import all required libraries

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
from sklearn.metrics import mean_squared_error,  mean_absolute_error
import scipy.stats

#Creating adjacency matrix
device = torch.device('cuda') if torch.cuda.is_available() else \
                                torch.device('cpu')

data_adj= pd.read_csv('/content/project-master/data/distanceNodes.txt', encoding='utf8', delimiter='\t')

#create new column calling Distance_KM, which calculates distance between nodes in km
data_adj['Distance_KM']=data_adj['NEAR_DIST']/1000

#  Change the nodes id
replacement_mapping_dict = {
    0: 141, 1: 143, 2: 195, 3: 181, 4: 5, 5: 88, 6: 138, 7: 254, 8: 142, 9: 113, 10: 192, 11: 243, 12: 78, 13: 92, 
    14: 177, 15: 126, 16: 228, 17: 47, 18: 220, 19: 72, 20: 281, 21: 323, 22:217, 23:296
}
data_fin = data_adj[["IN_FID", "NEAR_FID"]].replace(replacement_mapping_dict)

# Replace columns with the new created columns
data_adj['IN_FID'] = data_fin["IN_FID"]
data_adj['NEAR_FID'] = data_fin["NEAR_FID"]

### create Adjacency matrix and replace column and index names according to grid cells id where air quality stations are located

am = pd.DataFrame(np.zeros(shape=(24, 24)))
am.rename(columns=replacement_mapping_dict, index =replacement_mapping_dict, inplace=True)
adj_mat = am.sort_index(axis=1)
adj_mat_complete = adj_mat.sort_index()


### Adjacency matrix

for i in data_adj.IN_FID.unique():
  for j in data_adj.NEAR_FID.unique():
    if i==j:
      adj_mat_complete.at[i,j]=0
    else:      
      adj_mat_complete.at[i,j]=1/data_adj.loc[(data_adj['IN_FID'] == i) & (data_adj['NEAR_FID'] == j)]['Distance_KM']
   
      
print(adj_mat_complete)  

# set the seed in order to provide reproducibility of the code

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
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed_num)
  print(f"Random seed set as {seed_num}")


# read the file containing nodes (air quality monitoring stations) features as pandas dataframe
Mad_data_2019 = pd.read_csv('/content/project-master/data/Mad_Station_2019.csv')
Mad_data_2022 = pd.read_csv('/content/project-master/data/Mad_Station_2022.csv')

# to delete 'windDir' column as we will not need it for further analysis.
Mad_data_2019 = Mad_data_2019.drop(['windDir'], axis=1)
Mad_data_2022 = Mad_data_2022.drop(['windDir'], axis=1)


# convert dataframes to numpy arrays

fin_data=Mad_data_2019.to_numpy().reshape(-1, 24, 18)
fin_data_test=Mad_data_2022.to_numpy().reshape(-1, 24,18)
fin_data = fin_data[:100, :, :]
fin_data_test = fin_data_test[:100, :, :]
fin_data_test.shape

# the function to return data in original scale (reversing Z score)

def reverse_zscore(pandas_series, mean, std):
    '''Mean and standard deviation should be of original variable before standardisation'''
    yis=pandas_series*std+mean
    return yis





#to convert adjacency dataframe to numpy
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


# define A3T-GCN

#with tf.device('/device:GPU:0'):
#  set_seed(rnd_seed)

unit_num  = 256  # This is number of units in order to construct the architecture of the A3T-GCN. 

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                        out_channels=unit_num, 
                        periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(unit_num, periods)

    def forward(self, x, edge_index,  edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index,  edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

TemporalGNN(node_features=18, periods=12)

# the original part of the code to train the model with GPU support.

## GPU support
#device = torch.device('cpu') # cuda


## Create model and optimizers
#model = TemporalGNN(node_features=18, periods=12).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#model.train()

#print("Running training...")
#for epoch in range(1): 
#   loss = 0
#   step = 0
#    for snapshot in dataset:
#        snapshot = snapshot.to(device)
#        # Get model predictions
#        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
#        # Mean Squared Error
#        loss = loss + torch.mean((y_hat-snapshot.y)**2) 
#        step += 1
        

#    loss = loss / (step + 1)
#    loss.backward()
#    optimizer.step()
#    optimizer.zero_grad()
#    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))


device = torch.device('cpu') # cuda
model_loaded = TemporalGNN(node_features=18, periods=12).to(device)
load = torch.load('/content/project-master/data/model.pth', map_location=torch.device('cpu'))
model_loaded.load_state_dict(load)

model_loaded.eval()
loss = 0
step = 0

# Store for analysis
predictions = []
labels = []

for snapshot in dataset_test:
    snapshot = snapshot.to(device)
    # Get predictions
    y_hat = model_loaded(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    yhat_reverse = reverse_zscore(y_hat, means[0], stds[0])
    snapshot.y_reverse = reverse_zscore(snapshot.y, means[0], stds[0])
    # Root Mean Squared Error
    loss = loss + torch.sqrt(torch.mean((yhat_reverse-snapshot.y_reverse)**2)) 
    # Store for analysis below
    labels.append(snapshot.y_reverse)
    predictions.append(yhat_reverse)    
    step += 1

loss = loss / (step+1)
loss = loss.item()
print("Test RMSE: {:.4f}".format(loss))

'''model.eval()
loss = 0
step = 0

# Store for analysis
predictions = []
labels = []

for snapshot in dataset_test:
    snapshot = snapshot.to(device)
    # Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    yhat_reverse = reverse_zscore(y_hat, means[0], stds[0])
    snapshot.y_reverse = reverse_zscore(snapshot.y, means[0], stds[0])
    # Mean Absolute Error
    loss = loss + torch.mean(torch.abs(yhat_reverse-snapshot.y_reverse))
    # Store for analysis below
    labels.append(snapshot.y_reverse)
    predictions.append(yhat_reverse)
    step += 1
  

loss = loss / (step+1)
loss = loss.item()
print("Test MAE: {:.4f}".format(loss))'''

'''# it is calculated for all nodes

ALLNode_pred = []
ALLNode_true = []
for item in predictions:
  for node in range(24):
    for hour in range(12):
      ALLNode_pred.append(item[node][hour].detach().cpu().numpy().item(0))


for item in labels:
  for node in range(24):
    for hour in range(12):
      ALLNode_true.append(item[node][hour].detach().cpu().numpy().item(0))


ALLNode_pred_np = np.array(ALLNode_pred)
ALLNode_pred_np_resh = ALLNode_pred_np.reshape(-1, 24, 12)
ALLNode_true_np = np.array(ALLNode_true)
ALLNode_true_np_resh = ALLNode_true_np.reshape(-1, 24, 12)


scipy.stats.pearsonr(ALLNode_pred_np, ALLNode_true_np)'''





'''#to convert adjacency dataframe to numpy
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
'''

'''# standardise test data using means and standard deviation of train set

data_test = fin_data_test.transpose(
            (1, 2, 0)
        )
data_test = data_test.astype(np.float32)
data_test_norm= data_test- means.reshape(1, -1, 1)
data_test_norm= data_test_norm/ stds.reshape(1, -1, 1)

#to convert standardised test data to torch
data_test_norm = torch.from_numpy(data_test_norm)'''

#adj.shape

#data_test_norm.shape

'''#from adjacency matrix extract the edge indices and edge weights

edge_indices, values = dense_to_sparse(adj)
edge_indices = edge_indices.numpy()
values = values.numpy()
edges = edge_indices
edge_weights = values
batch =64'''

#edges.shape

#edge_weights.shape

'''class MadridDatasetLoader(object):
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
        time_steps_starter =   36 # it can be assigned as one of the following {0, 12, 24, 36}
        indices = [
            (i, i +time_steps_starter+ (num_timesteps_in + num_timesteps_out))
            for i in range(self.data_norm.shape[2] - (time_steps_starter+num_timesteps_in + num_timesteps_out) + 1)
        ]
        print(indices)
        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.data_norm[:, :, i : i + num_timesteps_in]).numpy().transpose(
            (2, 0, 1)
        ))
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

        return dataset'''

'''loader = MadridDatasetLoader(data_test_norm, edges, edge_weights, batch)
dataset_test = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
print("Dataset type:  ", dataset_test)'''


'''loader = MadridDatasetLoader(data_norm, edges, edge_weights, batch)
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
print("Dataset type:  ", dataset)'''


#next(iter(dataset))

'''# define A3T-GCN

with tf.device('/device:GPU:0'):
  set_seed(rnd_seed)

  unit_num  = 256 # This is number of units in order to construct the architecture of the TGCN. 
  class TemporalGNN(torch.nn.Module):
      def __init__(self, node_features):
          super(TemporalGNN, self).__init__()
          # Temporal Graph Convolutional Cell
          self.tgnn = TGCN2(in_channels=node_features, 
                            out_channels=unit_num, batch_size=12)
          # Equals single-shot prediction
          self.linear = torch.nn.Linear(unit_num, 12)

      def forward(self, x, edge_index,  edge_weight):
          """
          x = Node features for T time steps
          edge_index = Graph edge indices
          """
          h = self.tgnn(x, edge_index,  edge_weight)
          h = F.relu(h)
          h = self.linear(h)
          return h

TemporalGNN(node_features=18)'''

'''# GPU support
device = torch.device('cpu') # cuda


# Create model and optimizers
model = TemporalGNN(node_features=18).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
        loss = loss + torch.mean((y_hat-snapshot.y)**2) 
        step += 1
        

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))'''

'''# it is calculated for all nodes

ALLNode_pred = []
ALLNode_true = []
for item in predictions:
  for batch in range(12):
    for node in range(24):
      for hour in range(12):
        ALLNode_pred.append(item[batch][node][hour].detach().cpu().numpy().item(0))


for item in labels:
  for node in range(24):
    for hour in range(12):
      ALLNode_true.append(item[node][hour].detach().cpu().numpy().item(0))


ALLNode_pred_np = np.array(ALLNode_pred)
ALLNode_pred_np_resh = ALLNode_pred_np.reshape(-1, 12, 24, 12)
ALLNode_true_np = np.array(ALLNode_true)
ALLNode_true_np_resh = ALLNode_true_np.reshape(-1, 24, 12)'''

'''model.eval()
loss = 0
step = 0

# Store for analysis
predictions = []
labels = []

for snapshot in dataset_test:
    snapshot = snapshot.to(device)
    # Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    yhat_reverse = reverse_zscore(y_hat, means[0], stds[0])
    snapshot.y_reverse = reverse_zscore(snapshot.y, means[0], stds[0])
    # Mean Absolute Error
    loss = loss + torch.mean(torch.abs(yhat_reverse-snapshot.y_reverse))
    # Store for analysis below
    labels.append(snapshot.y_reverse)
    predictions.append(yhat_reverse)
    step += 1
  

loss = loss / (step+1)
loss = loss.item()
print("Test MAE: {:.4f}".format(loss))'''

'''ALLNode_pred_np_resh_Update = np.mean(ALLNode_pred_np_resh, axis=1)
true = ALLNode_true_np_resh.reshape(-1)
pred = ALLNode_pred_np_resh_Update.reshape(-1)
rmse = mean_squared_error(true, pred, squared=False)
mae = mean_absolute_error(true, pred)
scipy.stats.pearsonr(true, pred)'''





'''# standardise train data

data = fin_data.transpose(
            (1, 2, 0)
        )
data = data.astype(np.float32)

# standardise (via Z-Score Method)
means = np.mean(data, axis=(0, 2))
data_norm= data-means.reshape(1, -1, 1)
stds = np.std(data_norm, axis=(0, 2))
data_norm= data_norm/ stds.reshape(1, -1, 1)
fin_data_norm = data_norm.transpose(2, 0, 1)'''



#fin_data.shape

'''# standardise test data

data_test = fin_data_test.transpose(
            (1, 2, 0)
        )
data_test = data_test.astype(np.float32)
data_test_norm= data_test- means.reshape(1, -1, 1)
data_test_norm= data_test_norm/ stds.reshape(1, -1, 1)
fin_data_test_norm = data_test_norm.transpose(2, 0, 1)'''


#fin_data_test_norm.shape

'''# split dataset to X and y (dependent and independent)

def split_sequence(sequence, seq_notNorm, time_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
   
		# find the end of this pattern
		end_ix = i +12
		time_steps_starter = 0 # it can be assigned as one of the following {0, 12, 24, 36}

		# check if we are beyond the sequence
		if end_ix+time_steps_starter+time_steps > len(sequence)-1:
			break
		# gather input and output parts of the pattern    
		seq_x, seq_y = sequence[i:end_ix], seq_notNorm[end_ix+time_steps_starter:end_ix+time_steps_starter+time_steps]
		X.append(seq_x)
		y.append(seq_y)
    
	return np.array(X), np.array(y)
 

# choose a number of time steps 
time_steps = 12
X_train, y_train = split_sequence(fin_data_norm, fin_data_norm, time_steps)
X_test, y_test = split_sequence(fin_data_test_norm, fin_data_test_norm, time_steps)

# to select only nitrogen dioxide as a target feature
y_train = y_train[:, :, :, 0]
y_test = y_test[:, :, :, 0]'''


#y_test.shape

#X_train.shape

# reshape the data for LSTM input

'''number_selected_columns =18
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 24*number_selected_columns))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 24*number_selected_columns))'''


# define model

'''with tf.device('/device:GPU:0'):
  set_seed(rnd_seed)
  model = Sequential()
  model.add(Dense(432,input_shape=(X_train.shape[1], 24*number_selected_columns)))
  model.add(LSTM(512, return_sequences=True))
  model.add(Dense(24))
  model.compile(optimizer='adam', loss='mse')'''

# run LSTM

#lstm_model = model.fit(X_train_reshaped, y_train, epochs=1, verbose=2)


#y_test.shape

#yhat = model.predict(X_test_reshaped, verbose=1)

'''yhat_reverse = reverse_zscore(yhat, means[0], stds[0])
ytest_reverse = reverse_zscore(y_test, means[0], stds[0])
yhat_reshaped = yhat_reverse.reshape(-1,24)
y_test_reshaped= ytest_reverse.reshape(-1,24)
rmse = mean_squared_error(yhat_reshaped, y_test_reshaped, squared=False)
mae = mean_absolute_error(yhat_reshaped, y_test_reshaped)
print('Test Score: %.2f RMSE' % (rmse))
print('Test Score: %.2f MAE' % (mae))'''

#scipy.stats.pearsonr(yhat_reshaped.reshape(-1), y_test_reshaped.reshape(-1))





# standardise train data

'''data = fin_data.transpose(
            (1, 2, 0)
        )
data = data.astype(np.float32)

# standardise (via Z-Score Method)
means = np.mean(data, axis=(0, 2))
data_norm= data-means.reshape(1, -1, 1)
stds = np.std(data_norm, axis=(0, 2))
data_norm= data_norm/ stds.reshape(1, -1, 1)
fin_data_norm = data_norm.transpose(2, 0, 1)'''



#fin_data.shape

# standardise test data

'''data_test = fin_data_test.transpose(
            (1, 2, 0)
        )
data_test = data_test.astype(np.float32)
data_test_norm= data_test- means.reshape(1, -1, 1)
data_test_norm= data_test_norm/ stds.reshape(1, -1, 1)
fin_data_test_norm = data_test_norm.transpose(2, 0, 1)'''


#fin_data_test_norm.shape

# split dataset to X and y (dependent and independent)

'''def split_sequence(sequence, seq_notNorm, time_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
   
		# find the end of this pattern
		end_ix = i +12
		time_steps_starter = 0 # it can be assigned as one of the following {0, 12, 24, 36}

		# check if we are beyond the sequence
		if end_ix+time_steps_starter+time_steps > len(sequence)-1:
			break
		# gather input and output parts of the pattern    
		seq_x, seq_y = sequence[i:end_ix], seq_notNorm[end_ix+time_steps_starter:end_ix+time_steps_starter+time_steps]
		X.append(seq_x)
		y.append(seq_y)
    
	return np.array(X), np.array(y)
 

# choose a number of time steps 
time_steps = 12
X_train, y_train = split_sequence(fin_data_norm, fin_data_norm, time_steps)
X_test, y_test = split_sequence(fin_data_test_norm, fin_data_test_norm, time_steps)

# to select only nitrogen dioxide as a target feature
y_train = y_train[:, :, :, 0]
y_test = y_test[:, :, :, 0]'''


#y_test.shape

#X_train.shape


# reshape the data for GRU input

'''number_selected_columns =18
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 24*number_selected_columns))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 24*number_selected_columns))'''


# define model

'''with tf.device('/device:GPU:0'):
  set_seed(rnd_seed)
  model = Sequential()
  model.add(Dense(432,input_shape=(X_train.shape[1], 24*number_selected_columns)))
  model.add(GRU(512, return_sequences=True))
  model.add(Dense(24))
  model.compile(optimizer='adam', loss='mse')'''

# run GRU

#gru_model = model.fit(X_train_reshaped, y_train, epochs=1, verbose=2)


#yhat = model.predict(X_test_reshaped, verbose=1)

'''yhat_reverse = reverse_zscore(yhat, means[0], stds[0])
ytest_reverse = reverse_zscore(y_test, means[0], stds[0])
yhat_reshaped = yhat_reverse.reshape(-1,24)
y_test_reshaped= ytest_reverse.reshape(-1,24)
rmse = mean_squared_error(yhat_reshaped, y_test_reshaped, squared=False)
mae = mean_absolute_error(yhat_reshaped, y_test_reshaped)
print('Test Score: %.2f RMSE' % (rmse))
print('Test Score: %.2f MAE' % (mae))'''

#scipy.stats.pearsonr(yhat_reshaped.reshape(-1), y_test_reshaped.reshape(-1))



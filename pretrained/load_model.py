from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GATConv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2020) # seed for reproducible numbers

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
%matplotlib notebook

import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("framenet_v17")
from nltk.corpus import framenet as fn
import networkx as nx
import numpy as np

print("...creating networkx FN...")
G = nx.DiGraph()
for frame in fn.frames():
    G.add_node(frame.ID)
    for adj in frame.frameRelations:
        G.add_edge(adj.superFrame.ID, adj.subFrame.ID)
        G.add_edge(adj.subFrame.ID, adj.superFrame.ID)

# initialize frame embeddings with LASER sentence representations 
print("...embedding frames...")
!python -m laserembeddings download-models
from laserembeddings import Laser
laser = Laser()
sentences = [fn.frame(frameID).definition for frameID in G.nodes]
frame_embeddings = laser.embed_sentences(sentences, lang='en')

# convert networkx G into torch.geometric graph
print("...generating torch_geometric graph...")
# x = torch.from_numpy(np.array(G.nodes).reshape(-1, 1)).float()  # x.shape = (1221, 1)
x = torch.from_numpy(frame_embeddings)  # x.shape = (1221, 1024)
nodes_to_x = {node: i for i, node in enumerate(G.nodes)}  # map frame ID to index position in x
x_to_nodes = {i: node for i, node in enumerate(G.nodes)}  # reverse of nodes_to_x
edge_index = torch.Tensor(list(set([(nodes_to_x[src], nodes_to_x[tgt]) for src, tgt in G.edges]))).long()

data = Data(x=x, edge_index=edge_index.t().contiguous())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Load GAT Model
class NodeNorm(nn.Module):
    def __init__(self, unbiased=False, eps=1e-5):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x

class GAT(torch.nn.Module):
    def __init__(self, data, hid=109, hid2=256, in_head=9, out_head=10):
        super(GAT, self).__init__()
        self.hid = hid
        self.hid2 = hid2
        self.in_head = in_head
        self.out_head = out_head
        
        self.node_norm = NodeNorm()
        self.conv1 = GATConv(data.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid2, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data, training=True):
        x, edge_index = data.x, data.edge_index
        
        # DropEdge
        edge_index, _ = dropout_adj(data.edge_index, training=training)

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.
        x = nn.Dropout(p=0.4)(x)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.node_norm(x)
        x = nn.Dropout(p=0.4)(x)
        x = self.conv2(x, edge_index)
        x = self.node_norm(x)
        return x

if __name__ == "__main__":
    models = torch.load("model_and_embeddings.pt")
    outs = []
    for model in models:
        model.eval()
        outs.append(model(data, training=False))
    out = torch.mean(torch.stack(outs), dim=0)  # semantic frame embeddings

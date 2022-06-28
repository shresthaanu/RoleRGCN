import pandas as pd
import numpy as np
import os
import networkx as nx
from tqdm import tqdm 
from collections import Counter, defaultdict

import dgl
from dgl.nn import GraphConv,GATConv,SAGEConv,HeteroGraphConv


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


# Define a Heterograph Conv model
'''heterogeneous graph convolution module that first performs a separate graph convolution on each edge type, 
then sums the message aggregations on each edge type as the final result for all node types.'''

class RGCN_orig(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats_user, out_feats_news, out_feats_source, rel_names):
        super().__init__()

        # HeteroGraphConv takes in a dictionary of node types and node feature tensors as input, and returns another dictionary of node types and node features.
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
            
        self.user_layer = nn.Linear(hid_feats, out_feats_user)
        self.news_layer = nn.Linear(hid_feats, out_feats_news)
        self.source_layer = nn.Linear(hid_feats, out_feats_source)
        # self.follower_layer = nn.Linear(hid_feats, out_feats_user)

        self.lin = {'user':self.user_layer, 'news':self.news_layer, 'source':self.source_layer}
      

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h1 = h
        h = {k: self.lin[k](h[k]) for k, v in h.items()}
        return h, h1

        
class RGCN2_combine_losses(nn.Module):
    def __init__(self, in_feats_user,in_feats_news,in_feats_source, hid_feats, out_feats_user, out_feats_news, out_feats_source, rel_names):
        super().__init__()
        self.user_input1 = nn.Linear(in_feats_user, hid_feats)
        self.news_input1 = nn.Linear(in_feats_news, hid_feats)
        self.source_input1 = nn.Linear(in_feats_source, hid_feats)
        self.follower_input1 = nn.Linear(in_feats_user, hid_feats)
        self.user_input2 = nn.Linear(hid_feats, hid_feats)
        self.news_input2 = nn.Linear(hid_feats, hid_feats)
        self.source_input2 = nn.Linear(hid_feats, hid_feats)
        self.follower_input2 = nn.Linear(hid_feats, hid_feats)

        self.user_conv1 = HeteroGraphConv({ rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        self.news_conv1 = HeteroGraphConv({ rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        self.source_conv1 = HeteroGraphConv({rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        self.follower_conv1 = HeteroGraphConv({rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        
        self.user_conv2 = HeteroGraphConv({ rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        self.news_conv2 = HeteroGraphConv({ rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        self.source_conv2 = HeteroGraphConv({rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')
        self.follower_conv2 = HeteroGraphConv({rel: GraphConv(hid_feats, hid_feats) for rel in rel_names}, aggregate='mean')

        
        
        self.user_layer = nn.Linear(hid_feats, out_feats_user)
        self.news_layer = nn.Linear(hid_feats, out_feats_news)
        self.source_layer = nn.Linear(hid_feats, out_feats_source)
        self.follower_layer = nn.Linear(hid_feats, out_feats_user)
        self.user_layer2 = nn.Linear(hid_feats*6+in_feats_user, hid_feats)
        self.news_layer2 = nn.Linear(hid_feats*6+in_feats_news, hid_feats)
        self.source_layer2 = nn.Linear(hid_feats*6+in_feats_source, hid_feats)
        self.follower_layer2 = nn.Linear(hid_feats*6+in_feats_source, hid_feats)
        
        self.sigmoid_layer = nn.Sigmoid()
        
        self.input1 = {'user':self.user_input1, 'news':self.news_input1, 'source':self.source_input1, 'follower':self.follower_input1}
        self.input2 = {'user':self.user_input2, 'news':self.news_input2, 'source':self.source_input2, 'follower':self.follower_input2}
        self.conv1 = {'user':self.user_conv1, 'news':self.news_conv1, 'source':self.source_conv1, 'follower':self.follower_conv1}
        self.conv2 = {'user':self.user_conv2, 'news':self.news_conv2, 'source':self.source_conv2, 'follower':self.follower_conv2}
        self.lin1 = {'user':self.user_layer, 'news':self.news_layer, 'source':self.source_layer, 'follower':self.follower_layer}
        self.lin2 = {'user':self.user_layer2, 'news':self.news_layer2, 'source':self.source_layer2, 'follower':self.follower_layer2}
        # self.sig = {'user':self.sigmoid_layer, 'news':self.sigmoid_layer, 'source':self.sigmoid_layer, 'follower':self.sigmoid_layer}
        
       
        

    def forward(self, graph, inputs):
        # inputs are features of nodes
        l = [inputs] #collect node features and latent features generated in each layer to concat in linear layer
        h = {k: self.input1[k](inputs[k]) for k,v in inputs.items()}
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        l.append(h)
        h = {k: self.input2[k](h[k]) for k,v in h.items()}
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        l.append(h)
        h = {k: self.conv1[k](graph, h)[k] for k,v in h.items()}
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        l.append(h)
        h = {k: self.conv2[k](graph, h)[k] for k,v in h.items()}
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        l.append(h)
        h1 = h
        h = {k: self.lin1[k](h[k]) for k, v in h.items()}
        # h = {k: self.sigmoid_layer(v) for k, v in h.items()}
        return h, h1

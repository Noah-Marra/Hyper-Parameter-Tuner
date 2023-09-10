import DB
import Initialization
import UMAP_HDBSCAN
import sqlite3
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hdbscan
import umap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

from tqdm.notebook import trange
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from transformers import BertModel, BertTokenizer

#--------------------------------------------------------------------------------------------------------
#Load Data
sql_where = "WHERE IsValid=True AND QuestionID=1"
db_path = 'data/'
labelset_id = Initialization.init_labels(db_path)
text = DB.load_data(db_path + 'ConstructMapping.db', sql_where)
#text = DB.pre_process(text)
#--------------------------------------------------------------------------------------------------------
#Load Embedders
#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
use = hub.load(module_url)
#Sentence BERT
sbert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#BERT-Uncased
class bert_model(nn.Module):
    def __init__(self, model_name, freeze = False, device = 'cpu'):
        super().__init__()

        self.model = BertModel.from_pretrained(model_name)
        self.device = device

        if freeze:
            for layer in self.model.parameters():
                layer.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)
        # Obtain BERT embeddings
        with torch.no_grad():
            outputs = self.model(x['input_ids'])
            embeddings = outputs.last_hidden_state
            mean_pool = embeddings.sum(axis=1)

        return mean_pool
model_BERT = 'bert-base-uncased'
bert = bert_model(model_BERT, freeze=True)
#--------------------------------------------------------------------------------------------------------
#dataloader
def embed(model, model_type, text):
    final_embeddings=list()
    all_embeddings = []
    final_sentences = text

    batch_sz = 200 # batch_size
    for i in range(0, len(final_sentences), batch_sz):
        batch_sentences = final_sentences[i:i+batch_sz]
        for sent in batch_sentences:

            if model_type == 'use':
                tokens = tokenizer(sent, return_tensors='pt')
                embeddings = model(tokens)
                all_embeddings.extend(embeddings)

            elif model_type == 'bert':
                tokens = tokenizer(sent, return_tensors='pt')
                embeddings = model(tokens)
                final_embeddings.extend(embeddings)
                all_embeddings = torch.stack(final_embeddings)

            elif model_type == 'sbert':
                embeddings = model.encode(sent)
                all_embeddings.append(embeddings)

    return all_embeddings

#--------------------------------------------------------------------------------------------------------
#Retrieve Embeddings
#use_embeddings = embed(use, 'use', text)
#bert_embeddings = embed(bert, 'bert', text)
sbert_embeddings = embed(sbert,'sbert', text)
#--------------------------------------------------------------------------------------------------------
hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(5,25)),
    "n_components": hp.choice('n_components', range(5,25)),
    "min_cluster_size": hp.choice('min_cluster_size', range(2,25)),
    "random_state": 15
}
label_lower = 18
label_upper = 70
max_evals = 30
#--------------------------------------------------------------------------------------------------------
#best_use_params, best_use_clusters, use_trials, use_probabilities = UMAP_HDBSCAN.bayesian_search(use_embeddings,space=hspace,label_lower=label_lower,
#                                                                label_upper=label_upper,max_evals=max_evals)
#best_bert_params, best_bert_clusters, bert_trials, bert_probabilities = UMAP_HDBSCAN.bayesian_search(bert_embeddings,space=hspace,label_lower=label_lower,
#                                                                label_upper=label_upper,max_evals=max_evals)
best_sbert_params, best_sbert_clusters, sbert_trials, sbert_probabilities = UMAP_HDBSCAN.bayesian_search(sbert_embeddings,space=hspace,label_lower=label_lower,
                                                                label_upper=label_upper,max_evals=max_evals)
#--------------------------------------------------------------------------------------------------------
#Dimensionality Reduction
"""
pca = PCA(n_components=3)
pca.fit(all_embeddings)
transformed = pca.transform(all_embeddings)
"""
#----------------------------------------------------------------------------------
"""
#Cluster Algorithm
K_range = np.arange(0.1,1,0.1)
inertia_values = []
best_score = -1
score = []
silhouette = []

for k in K_range:
    clusterer = KMeans(n_clusters=k)
    clustered_text = clusterer.fit_predict(all_embeddings.cpu())
    labels = clusterer.labels_

    #inertia_values.append(clusterer.inertia_)
    score = silhouette_score(all_embeddings, labels)

    # Identify the Silhouette Score for Optimal K-Value
    if score > best_score:
        best_score = score
        optimal = k

    silhouette.append(score)


print(silhouette)
"""
#--------------------------------------------------------------------------------------------------------
#Identify the Elbow Point for Optimal K-Value
"""elbow_point = 0
for i in range(1, len(inertia_values)-1):
    if inertia_values[i] - inertia_values[i+1] < 0.2* (inertia_values[i-1] - inertia_values[i]):
        elbow_point = i + 1
        break


#--------------------------------------------------------------------------------------------------------
print(f"The Optimal eps is: {optimal}")

#Visualize Inertia Values
plt.plot(K_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.show()
"""
#--------------------------------------------------------------------------------------------------------

'''
k=100
clusterer_optimal = KMeans(n_clusters=k)
clustered_text_optimal = clusterer_optimal.fit_predict(all_embeddings.cpu())
labels = clusterer_optimal.labels_
print(labels)
print(labels.shape)
'''
#--------------------------------------------------------------------------------------------------------

'''
#Graphical Representation
# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D
ax.scatter(all_embeddings[:, 0], all_embeddings[:, 1], all_embeddings[:, 2], c=labels, cmap='viridis')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()
'''
#--------------------------------------------------------------------------------------------------------

'''
#create class for model
class model(nn.Module):
    def __init__(self, model_name, freeze = False, device = 'cpu'):
        super().__init__()

        self.model =
        self.device = device

        if freeze:
            for layer in self.model.parameters():
                layer.requires_grad = False

    def forward(self, x):
        x = x.to(self.device)
        # Obtain BERT embeddings
        with torch.no_grad():
            outputs = self.model(x['input_ids'])
            embeddings = outputs.last_hidden_state
            mean_pool = embeddings.sum(axis=1)

        return mean_pool
        '''


import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
boxes = torch.load("tokens_specialization/box_predictions.pt")
embeddings = torch.load("tokens_specialization/embeddings.pt")
rel_boxes = torch.load("tokens_specialization/rel_box_predictions.pt")
pca = PCA(n_components=2)
numpy_embeddings_for_pca = embeddings.flatten(0,1).numpy()
pca_embeddings = pca.fit_transform(numpy_embeddings_for_pca)
for t in range(25):
    relevant_pca_embeddings = pca_embeddings[t*4992:(t+1)*4992,:]
    x = relevant_pca_embeddings[:,0]
    y = relevant_pca_embeddings[:,1]
    plt.scatter(x,y,s=1)
    plt.savefig("tokens_specialization/token_embeddings_" + str(t) + ".jpg")
    plt.close()



for t in range(boxes.shape[0]):
    token = boxes[t]
    xc = token[:,0].numpy()
    yc = token[:,1].numpy()
    plt.scatter(xc,yc,s=1)
    plt.savefig("tokens_specialization/token_bb_" + str(t) + ".jpg")
#scatter xc,yc 

plt.close()

for t in range(rel_boxes.shape[0]):
    token = rel_boxes[t]
    xc = token[:,0].numpy()
    yc = token[:,1].numpy()
    plt.scatter(xc,yc,s=1)
    plt.savefig("tokens_specialization/token_bb_rel_" + str(t) + ".jpg")
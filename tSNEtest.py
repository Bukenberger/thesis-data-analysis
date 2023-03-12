from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
## Loading and curating the data

RECORDING_FOLDER_PATH = "./Recordings/"

def label_saccades( eye_data_json, supervised_saccades_txt, id ) -> pd.DataFrame:
    df = pd.read_json( f'{RECORDING_FOLDER_PATH}{eye_data_json}' )
    df.insert( 0, "id", id )
    df.insert( 0, "isSaccade", 0 )
    txt = open( f'{RECORDING_FOLDER_PATH}{supervised_saccades_txt}' )
    for line in txt:
        line = line.rstrip().split(':')
        lower_bound = int( line[0] )
        upper_bound = int( line[1] )
        df.loc[ (df.microTimestamp >= lower_bound) & (df.microTimestamp <= upper_bound), 'isSaccade' ] = 1
    return df

df1 = label_saccades( "jess-recording-1.json", "jess-saccades-1.txt", 0 )
df2 = label_saccades( "jess-recording-2.json", "jess-saccades-2.txt", 0 )
df3 = label_saccades( "teran-recording-1.json", "teran-saccades-1.txt", 1 )
df4 = label_saccades( "luke-recording-1.json", "luke-saccades-1.txt", 2 )

# frames = [ df1, df2, df3, df4 ]
frames = [ df3 ]
df = pd.concat( frames )
saccade_list = df['isSaccade']

df = df.drop(
        columns=[
            'systemTimestamp',
            'deviceTimestamp',
            'microTimestamp',
            'milliTimestamp',
            'secTimestamp',
            'minTimestamp',
            "isSaccade",
            "id",
            'leftGazeRayIsValid',
            'leftEyeIsBlinking',
            'leftPupilDiameterIsValid',
            'leftPositionGuideIsValid',
            'rightGazeRayIsValid',
            'rightEyeIsBlinking',
            'rightPupilDiameterIsValid',
            'rightPositionGuideIsValid',
            'gazeRayIsValid',
            'convergenceDistanceIsValid'
        ]
    )

# digits = datasets.load_digits(n_class=10)
X = df
y = saccade_list
print(X)
print(y)
n_samples, n_features = X.shape
## Function to Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], 
                 X[i, 1], 
                #  str(saccade_list[i]),
                'o' if saccade_list[i] == 0 else 'x',
                 color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
## Computing PCA
print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig('PCA_PLOT.png', dpi=300)
## Computing t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig('t-SNE_PLOT.png', dpi=300)
# plt.show()
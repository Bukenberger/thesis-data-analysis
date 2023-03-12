from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

RECORDING_FOLDER_PATH = "./Recordings/"

def label_saccades( eye_data_json, supervised_saccades_txt, id ) -> pd.DataFrame:
    df = pd.read_json( f'{RECORDING_FOLDER_PATH}{eye_data_json}' )
    df.insert( 0, "id", id )
    df.insert( 0, 'movementType', 'non-saccade' )
    txt = open( f'{RECORDING_FOLDER_PATH}{supervised_saccades_txt}' )
    for line in txt:
        line = line.rstrip().split(':')
        lower_bound = int( line[0] )
        upper_bound = int( line[1] )
        df.loc[ (df.microTimestamp >= lower_bound) & (df.microTimestamp <= upper_bound), 'movementType' ] = 'saccade'
    return df

df1 = label_saccades( "jess-recording-1.json", "jess-saccades-1.txt", 0 )
df2 = label_saccades( "jess-recording-2.json", "jess-saccades-2.txt", 0 )
df3 = label_saccades( "teran-recording-1.json", "teran-saccades-1.txt", 1 )
df4 = label_saccades( "luke-recording-1.json", "luke-saccades-1.txt", 2 )

frames = [ df1, df2, df3, df4 ]
# frames = [ df2 ]
df = pd.concat( frames )
saccade_list = df['movementType']

# remove non-numeric and time-related columns
df = df.drop(
        columns=[
            'systemTimestamp',
            'deviceTimestamp',
            'microTimestamp',
            'milliTimestamp',
            'secTimestamp',
            'minTimestamp',
            "movementType",
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

pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2)),])

pca_data = pd.DataFrame(
    pipeline.fit_transform(df),
    columns=["PC1", "PC2"],
    index=df.index,
)

pca_data['movementType'] = saccade_list

print(pipeline.steps)

pca_step = pipeline.steps[1][1]
loadings = pd.DataFrame(
    pca_step.components_.T,
    columns=["PC1", "PC2"],
    index=df.columns,
)
print( loadings )

def loading_plot(
    coeff, labels, scale=1, colors=None, visible=None, ax=plt, arrow_size=0.5
):
    for i, label in enumerate(labels):
        if visible is None or visible[i]:
            ax.arrow(
                0,
                0,
                coeff[i, 0] * scale,
                coeff[i, 1] * scale,
                head_width=arrow_size * scale,
                head_length=arrow_size * scale,
                color="#000" if colors is None else colors[i],
            )
            ax.text(
                coeff[i, 0] * 1.15 * scale,
                coeff[i, 1] * 1.15 * scale,
                label,
                color="#000" if colors is None else colors[i],
                ha="center",
                va="center",
            )

g = sns.scatterplot(data=loadings, x="PC1", y="PC2", hue=df.columns)

df_st = StandardScaler().fit_transform(df)
# print(pd.DataFrame(df_st, columns=df.columns).head(2))

# Add loadings
# loading_plot(loadings[["PC1", "PC2"]].values, loadings.index, scale=2, arrow_size=0.08)

x = pca_step.explained_variance_ratio_[0]
y = pca_step.explained_variance_ratio_[1]

# Add variance explained
g.set_xlabel(f"PC1 ({x*100:.2f} %)")
g.set_ylabel(f"PC2 ({y*100:.2f} %)")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.3))

plt.savefig("2D_PCA_with_loadings.png", bbox_inches='tight', dpi=200)

# df_st = StandardScaler().fit_transform(df)
# # print(pd.DataFrame(df_st, columns=df.columns).head(2))

# pca_out = PCA().fit(df_st)

# get the component variance
# Proportion of variance (from PC1 to PC25)
# print(pca_out.explained_variance_ratio_)

# Cumulative proportion of variance (from PC1 to PC25)
# print(np.cumsum(pca_out.explained_variance_ratio_))

# component loadings or weights (correlation coefficient between original variables and the component)
# component loadings represents thee elements of the eigenvector
# the squared loadings within the PCs always sums to 1
# loadings = pca_out.components_
# num_pc = pca_out.n_features_in_
# pc_list= [f'PC{str(i)}' for i in list(range(1, num_pc + 1))]
# loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
# loadings_df['variable'] = df.columns.values
# print(loadings_df)

# positive and negative values in component loadings reflects the positive and negative
# correlation of the variables with the PCs. Except A and B, all other variables have
# positive projection on the first PC.

# get correlation matrix plot for loadings
# import seaborn as sns
# import matplotlib.pyplot as plt
# ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
# plt.show()

# get eigenvalues (variance explained by each PC)
# pca_out.explained_variance_

# # get scree plot (for screen or elbow test)
# from bioinfokit.visuz import cluster
# cluster.screeplot(obj=[pc_list, pca_out.explained_variance_ratio_])

# # get PCA loadings plots
# # 2D
# cluster.pcaplot(
#         x=loadings[0], 
#         y=loadings[1], 
#         labels=df.columns.values,
#         var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
#         var2=round(pca_out.explained_variance_ratio_[1]*100, 2)
#     )

# # 3D
# cluster.pcaplot(
#         x=loadings[0], 
#         y=loadings[1], 
#         z=loadings[2],
#         labels=df.columns.values,
#         var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
#         var2=round(pca_out.explained_variance_ratio_[1]*100, 2),
#         var3=round(pca_out.explained_variance_ratio_[2]*100, 2)
#     )

# PCA Biplot
# get PC scores
# pca_score = PCA().fit_transform(df_st)

# get 2D biplot
# cluster.biplot(
#     cscore=pca_score,
#     loadings=loadings,
#     labels=df.columns.values,
#     var1=round(pca_out.explained_variance_ratio_[0]*100,2),
#     var2=round(pca_out.explained_variance_ratio_[1]*100,2)
# )

# # get 3D biplot
# cluster.biplot(
#     cscore=pca_score,
#     loadings=loadings,
#     labels=df.columns.values,
#     var1=round(pca_out.explained_variance_ratio_[0]*100,2),
#     var2=round(pca_out.explained_variance_ratio_[1]*100,2),
#     var3=round(pca_out.explained_variance_ratio_[2]*100,2)
# )
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

frames = [ df1, df2, df3, df4 ]
# frames = [ df2 ]
df = pd.concat( frames )
saccade_list = df['isSaccade']

# remove non-numeric and time-related columns
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

# print(df.head(2))
print(df)

# Train and Test Data Seperation
X_train, X_test, Y_train, Y_test = train_test_split(df, saccade_list, test_size=0.25, 
                                                    stratify=saccade_list, random_state=30)

print ("train feature shape: ", X_train.shape)
print ("test feature shape: ", X_test.shape)

# For PCA, First Need to Scale the Data.  
scaler1 = StandardScaler()
scaler1.fit(df)
feature_scaled = scaler1.transform(df)

# Now Apply PCA
pca1 = PCA(n_components=4)
pca1.fit(feature_scaled)
feature_scaled_pca = pca1.transform(feature_scaled)
print("shape of the scaled and 'PCA'ed features: ", np.shape(feature_scaled_pca))

# Let's see the variance to see out of the 
# 4 components which are contributing most 

feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))

print ("Variance Ratio of the 4 Principal Components Ananlysis: ", feat_var_rat)

# #print (type(cancer.target))
# saccade_list = df['isSaccade'].tolist()
# print (type(saccade_list))
#print (saccade_list)
#print (type(yl))
feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]
feature_scaled_pca_X2 = feature_scaled_pca[:, 2]
feature_scaled_pca_X3 = feature_scaled_pca[:, 3]

labels = saccade_list
colordict = {0:'brown', 1:'darkslategray'}
piclabel = {0:'Not Saccade', 1:'Saccade'}
markers = {0:'o', 1:'*'}
alphas = {0:0.3, 1:0.4}

fig = plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
for l in np.unique(labels):
    ix = np.where(labels==l)
    plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l], 
               label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Second Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.subplot(1,2,2)
for l1 in np.unique(labels):
    ix1 = np.where(labels==l1)
    plt.scatter(feature_scaled_pca_X2[ix1], feature_scaled_pca_X3[ix1], c=colordict[l1], 
               label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
plt.xlabel("Third Principal Component", fontsize=15)
plt.ylabel("Fourth Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.savefig('Saccade_labels_PCAs.png', dpi=200)
# plt.show()

# Pipeline Steps are StandardScaler, PCA and SVM 
pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]

check_params= {
    'pca__n_components': [2], 
    'SupVM__C': [0.1, 0.5, 1, 10,30, 40, 50, 75, 100, 500, 1000], 
    'SupVM__gamma' : [0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}

pipeline = Pipeline(pipe_steps)

from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

print ("Start Fitting Training Data")
for cv in tqdm(range(4,6)):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train, Y_train)
    print ("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test, Y_test)))
    print ("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
    print (create_grid.best_params_)

print ("out of the loop")

print ("grid best params: ", create_grid.best_params_) 
# use the best one

# Time for Prediction and Plotting Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

Y_pred = create_grid.predict(X_test)
# print (Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: \n")
print(cm)


df_cm = pd.DataFrame(cm, range(2), range(2))

sns.heatmap(df_cm, annot=True, cbar=False)
plt.title("Confusion Matrix", fontsize=14)
plt.savefig("Confusion Matrix.png", dpi=200)

# 2D Decision Boundary

scaler1 = StandardScaler()
scaler1.fit(X_test)
X_test_scaled = scaler1.transform(X_test)


pca2 = PCA(n_components=2)
X_test_scaled_reduced = pca2.fit_transform(X_test_scaled)


# svm_model = SVC(kernel='rbf', C=float(create_grid.best_params_['SupVM__C']), 
#                 gamma=float(create_grid.best_params_['SupVM__gamma']))

svm_model = SVC(kernel='rbf', C=1., gamma=0.5)

classify = svm_model.fit(X_test_scaled_reduced, Y_test)

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print ('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print ('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))#,
                         #np.arange(z_min, z_max, h))
    return xx, yy

X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12,9))
fig.patch.set_facecolor('white')
cdict1={0:'lime',1:'deeppink'}

Y_tar_list = Y_test.tolist()
yl1= [int(target1) for target1 in Y_tar_list]
labels1=yl1
 
labl1={0:'Saccade',1:'Not Saccade'}
marker1={0:'*',1:'d'}
alpha1={0:.8, 1:0.5}

for l1 in np.unique(labels1):
    ix1=np.where(labels1==l1)
    ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
           edgecolors='navy', label='Support Vectors')

plot_contours(ax, classify, xx, yy,cmap='seismic', alpha=0.4)
plt.legend(fontsize=15)

plt.xlabel("1st Principal Component",fontsize=14)
plt.ylabel("2nd Principal Component",fontsize=14)

plt.savefig('ClassifySaccade_NotSaccade2D_Decs_FunctG10.png', dpi=300)
# plt.show()
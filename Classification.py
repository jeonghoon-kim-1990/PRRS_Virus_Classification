import pandas as pd 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support


from sklearn import svm
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import networkx as nx

import seaborn as sns
import scipy.stats as stats

from numpy import linalg as LA

import time
import random

from mpl_toolkits import mplot3d


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import train_test_split


def mlv_data(df):
    """function extracting part of mlv """
    mlv = df[df['Seq_Comment_Result'] == 'Ingelvac MLV like']
    return mlv

def wild_data(df):
    """function extracting part of wild """
    wild = df[df['Seq_Comment_Result'] == 'Wild type']
    return wild


def mlv_wild_data(df):
    """function extracting part of result (mlv + wild) """
    mlv = df[df['Seq_Comment_Result'] == 'Ingelvac MLV like']
    print('# of mlv like: {}'.format(len(mlv)))
    wild = df[df['Seq_Comment_Result'] == 'Wild type']
    print('# of wild type: {}'.format(len(wild)))
    df1 = mlv.append(wild)
    return df1

def char_to_num(char):
    """
    Five factor solution score 
    for the 54 selected amino acid attributes
    
    :Input: Character(Amino acid attribute) 
    :Output: 1x5 float type Array based on the conversion table
    """
    dic = {
        'A': np.array([-0.591, -1.302, -0.733, 1.570, -0.146]),
        'C': np.array([-1.343, 0.465, -0.862, -1.020, -0.255]),
        'D': np.array([1.050, 0.302, -3.656, -0.259, -3.242]),
        'E': np.array([1.357, -1.453, 1.477, 0.113, -0.837]),
        'F': np.array([-1.006, -0.590, 1.891, -0.397, 0.412]),
        'G': np.array([-0.384, 1.652, 1.330, 1.045, 2.064]),
        'H': np.array([0.336, -0.417, -1.673, -1.474, -0.078]),
        'I': np.array([-1.239, -0.547, 2.131, 0.393, 0.816]),
        'K': np.array([1.831, -0.561, 0.533, -0.277, 1.648]),
        'L': np.array([-1.019, -0.987, -1.505, 1.266, -0.912]),
        'M': np.array([-0.663, -1.524, 2.219, -1.005, 1.212]),
        'N': np.array([0.945, 0.828, 1.299, -0.169, 0.933]),
        'P': np.array([0.189, 2.081, -1.628, 0.421, -1.392]),
        'Q': np.array([0.931, -0.179, -3.005, -0.503, -1.853]),
        'R': np.array([1.538, -0.055, 1.502, 0.440, 2.897]),
        'S': np.array([-0.228, 1.399, -4.760, 0.670, -2.647]),
        'T': np.array([-0.032, 0.326, 2.213, 0.908, 1.313]),
        'V': np.array([-1.337, -0.279, -0.544, 1.242, -1.262]),
        'W': np.array([-0.595, 0.009, 0.672, -2.128, -0.184]),
        'Y': np.array([0.260, 0.830, 3.097, -0.838, 1.512]),
        '?': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        'X': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    }
    
    if char in dic:
        num = dic.get(char)
    else:
        num = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        print('{} is not in the table'.format(char))
    return num

def aacode_to_mat(aacode):
    """
    Amino acid code to matrix function.
    Input: List of characters (Amino acid)
    Output: appened array
    Each character will change to 1x5 matrix and append all arrays.
    """
    arr=[]
    for a in aacode:
        arr.append(char_to_num(a))
    aacode  = arr
    return arr

def aacode_to_mat2(aacode):
    """
    Amino acid code to matrix function.
    Input: List of characters (Amino acid)
    Output: appened array
    Each character will change to 1x5 matrix and append all arrays.
    """
    arr=[]
    for a in aacode:
        arr = np.append(arr,char_to_num(a))
    aacode  = arr
    return arr

def clean_columns(df):
    """Clean columns name"""
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    return df.columns

def reindex(df):
    """Reindexing columns in dataframe"""
    df = df.reset_index(inplace=False)
    df = df.drop(columns = ['index'])
    return df

def train_test_split_made(X,Y,split_pr = 0.8):
    """train-test split"""
    n = len(X)
    Ind = np.arange(n) 
    np.random.shuffle(Ind) 
    train_size = int(split_pr * n) # set training set size
    X_tr, X_te = X[Ind[:train_size]], X[Ind[train_size:]]
    Y_tr, Y_te = Y[Ind[:train_size]], Y[Ind[train_size:]]
    return (X_tr,Y_tr), (X_te, Y_te)

def multi_plot(df,labels):
    """ function to plot 2D points with multiple labels"""
    ax = sns.relplot(
    data=df[labels],
    x=labels[0], y=labels[1],
    hue = labels[2],
    legend = "brief",
    style = labels[3],
    size = 4,
    sizes=(20, 200),
    kind = 'scatter',
    height=8.27, aspect=11.7/8.27
)
    return plot.show(ax)

def column_selection(df,thres, score_list):
    colu = [True if i>thres else False for i in score_list]
    colcol = np.repeat(colu,5).tolist()
    dfl = df.iloc[:,colcol]
    return dfl

def visualize_score(score_list, thres_list=[]):
    num = len(score_list)
    ind2 = np.arange(num)
    fig = plt.figure(figsize=(20,10))
    aimp2 = np.sort(score_list, axis=None)
    len_ind = len(score_list)
                  
    for i in range(len(thres_list)):
        tthres = np.reshape(thres_list*num, (num, len(thres_list)))
        plt.plot(ind2, tthres[:,i], 'r')
    
    plt.bar(ind2, aimp2)
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance Score')
    plt.title("Visualizing Important Features with location")
    plt.legend()
    plt.show()
       
def visualize_score_part(score_list, thres_list=[], cutoff=None):
    fig = plt.figure(figsize=(18,20))
    n_score_list = score_list[score_list>cutoff]
    aimp2 = np.sort(n_score_list, axis=None)
    len_ind = len(n_score_list)
    ind_imp = np.argsort(score_list)[-len_ind:]
    ind2 = np.arange(len_ind)
                  
    for i in range(len(thres_list)):
        tthres = np.reshape(thres_list*len_ind, (len_ind, len(thres_list)))
        plt.plot(ind2, tthres[:,i], 'r')
    
                  
    ind2 = np.arange(len_ind)
    plt.bar(ind2, aimp2)
    plt.xticks(ind2,ind_imp+1
               ,  rotation=-45)
    plt.xlabel('AA position')
    plt.ylabel('Feature Importance Score')
    plt.title("Visualizing Important Features with location")
    plt.legend()
    plt.show()
    
def FI_RF_experiment(x,y, test_size=0.33):
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=test_size)#, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_tr, y_tr)
    y_pred_train = rf.predict(x_tr) # Predictions on training
    y_pred_test = rf.predict(x_te) # predictions for test
    preds = rf.predict(x_te)
    
    y_prob = rf.predict_proba(x_te)
    y_prob = y_prob[:,1]

    auc = roc_auc_score(y_te,y_prob)
    acc = accuracy_score(y_te,preds)
    
    feature_imp = pd.Series(rf.feature_importances_)
    imp = feature_imp.values.reshape(200,5)
    imp2 = imp.sum(axis=1)
    return imp2, feature_imp
    #print("Accuracy:", accuracy_score(y_te,preds))
    
def RF_experiment(x,y, test_size=0.33):
    t= time.time()
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=test_size)#, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_tr, y_tr)
    y_pred_train = rf.predict(x_tr) # Predictions on training
    y_pred_test = rf.predict(x_te) # predictions for test
    preds = rf.predict(x_te)
    
    y_prob = rf.predict_proba(x_te)
    y_prob = y_prob[:,1]

    auc = roc_auc_score(y_te,y_prob)
    acc = accuracy_score(y_te,preds)
    elapsed = time.time() - t
    prf = precision_recall_fscore_support(y_te, preds, average = 'binary')
    return acc, auc, elapsed, prf[0], prf[1], prf[2]
    #print("Accuracy:", accuracy_score(y_te,preds))
                  
                  
def SVM_experiment(x,y, test_size=0.33):
    t = time.time()
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=test_size)#,random_state=42)
    clf = svm.SVC(kernel = 'linear',  probability=True)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    #print("Accuracy:",metrics.accuracy_score(y_te, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    #print("Precision:",metrics.precision_score(y_te, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    #print("Recall:",metrics.recall_score(y_te, y_pred))

    y_prob = clf.predict_proba(x_te)
    y_prob = y_prob[:,1]
    
    acc = metrics.accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te,y_prob)
    #print("AUC: {}".format(auc))
    elapsed = time.time() - t
    prf = precision_recall_fscore_support(y_te, y_pred, average = 'binary')
    return acc, auc, elapsed, prf[0], prf[1], prf[2]

def KNN_experiment(x,y,k=5, test_size=0.33):
    t = time.time()
    x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=test_size)#, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_tr, y_tr)
    y_pred = knn.predict(x_te)

    y_prob = knn.predict_proba(x_te)
    y_prob = y_prob[:,1]
    
    acc = metrics.accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te,y_prob)
    elapsed = time.time() - t
    prf = precision_recall_fscore_support(y_te, y_pred, average = 'binary')

    return acc, auc, elapsed, prf[0], prf[1], prf[2]

def auc_eval(y_true, y_pred):
    auc = tf.keras.metrics.AUC(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def CNN_experiment(x,y, test_size=0.33):
    t = time.time()
    x_image = np.zeros((len(x), 200, 5))
    for i in range(len(x)):
        x_image[i][:][:] = np.resize(x.values[i].tolist(),(200,5))
    
    x_train, x_test, y_train, y_test = \
    train_test_split(x_image,y,test_size = test_size, random_state = 42)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(200, 5)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=[tf.keras.metrics.AUC()])    
    model.fit(x_train, y_train, epochs=3)

    test_loss, test_acc = model.evaluate(x_test,y_test, verbose=2)
    y_pred = model.predict_classes(x_test)
    
    preds = model.predict_proba(x_test)
    
    
    print('\n Accuracy:', test_acc)
    acc = test_acc
    elapsed = time.time() - t
    
    

    
    y_te = y_test
    #auc = auc_eval(y_te, preds)
    auc = roc_auc_score(y_te,preds[:,1])
    prf = precision_recall_fscore_support(y_te, y_pred, average = 'binary')

    return acc, auc, elapsed, prf[0], prf[1], prf[2]

def get_dis_sim(df_cor):
    u_dis = []
    AA_sim = []
    df_cor = reindex(df_cor)
    df_cor['Longitude'][1] - df_cor['Latitude'][2]
    #df_cor['AA'][0].shape()
    #- df_cor['AA'][1].tolist()
    AA_mat = df_cor.iloc[:,-1:]
    AA_image = np.zeros((len(AA_mat), 200, 5))
    for i in range(len(AA_image)):
        AA_image[i][:][:] = AA_mat.iloc[i,0]
    
    for i in range(len(df_cor)):
        for j in range(i,len(df_cor)):
            dx = df_cor['Longitude'][j] - df_cor['Longitude'][i] 
            dy = df_cor['Latitude'][j] - df_cor['Latitude'][i]
            dis = np.sqrt(dx**2 + dy**2)
            sim = LA.norm(AA_image[i][:][:] - AA_image[j][:][:],2)
            if (dis != 0.0) and (sim !=0.0):
                u_dis.append(dis)
                AA_sim.append(1/sim)
    
    d = {'dis': u_dis, 'sim':AA_sim}
    df_dis_sim = pd.DataFrame(data = d)
    #print(u_dis)
    #print(AA_sim)
    return df_dis_sim, u_dis, AA_sim


def get_nodes_edges(df4):
    t = time.time()
    """
    function to generate nodes and edges
    =input= df4 has to have three columns
    La, Long, AA, and also, AA has to be 200x5 matrix
    After df4.AA = [aacode_to_mat(aacode) for aacode in df4.AA]
    """
    x_y_ = df4[['Longitude','Latitude']].values
    n_nodes = np.arange(len(df4))
    position = dict(zip(n_nodes, x_y_)) 

    AA_mat = df4.iloc[:,-1:]
    AA_image = np.zeros((len(AA_mat), 200, 5))

    for i in range(len(AA_image)):
        AA_image[i][:][:] = AA_mat.iloc[i,0]

    G = nx.Graph()
    cutoff = 1
    eps = 0.001
    for i in range(len(df4)):
            for j in range(i, len(df4)):
                l2 = LA.norm(AA_image[i][:][:]-AA_image[j][:][:],2)
                w = -l2
                #w = 1/eps if l2==0 else 1/l2
                G.add_edge(i,j,weight = w)
                #G.add_edge(i,j,weight = w if w >= cutoff else 0)
    elapsed = time.time() - t
    print(elapsed)
    
    return G, position


def get_nodes_edges_binary(df4):
    t = time.time()
    """
    function to generate nodes and edges
    =input= df4 has to have three columns
    La, Long, AA, and also, AA has to be 200x5 matrix
    After df4.AA = [aacode_to_mat(aacode) for aacode in df4.AA]
    """
    x_y_ = df4[['Longitude','Latitude']].values
    n_nodes = np.arange(len(df4))
    position = dict(zip(n_nodes, x_y_)) 

    AA_mat = df4.iloc[:,-1:]
    AA_image = np.zeros((len(AA_mat), 200, 5))

    for i in range(len(AA_image)):
        AA_image[i][:][:] = AA_mat.iloc[i,0]

    G = nx.Graph()
    cutoff = 1
    eps = 0.001
    for i in range(len(df4)):
            for j in range(i, len(df4)):
                l2 = LA.norm(AA_image[i][:][:]-AA_image[j][:][:],2)
                if l2>0.0:
                    w = 1.0
                else:
                    w = 0.0
                #w = 1/eps if l2==0 else 1/l2
                G.add_edge(i,j)#,weight = w)
                #G.add_edge(i,j,weight = w if w >= cutoff else 0)
    elapsed = time.time() - t
    print(elapsed)
    
    return G, position

def get_nodes_edges_3d(df4):
    t = time.time()
    """
    function to generate nodes and edges
    =input= df4 has to have three columns
    La, Long, AA, and also, AA has to be 200x5 matrix
    After df4.AA = [aacode_to_mat(aacode) for aacode in df4.AA]
    """
    x_y_z_ = df4[['Longitude','Latitude','Height']].values
    n_nodes = np.arange(len(df4))
    position = dict(zip(n_nodes, x_y_z_)) 

    AA_mat = df4.AA.values
    AA_image = np.zeros((len(AA_mat), 200, 5))

    for i in range(len(AA_image)):
        AA_image[i][:][:] = AA_mat[i].reshape(200,5)

    G = nx.Graph()
    cutoff = 1
    eps = 0.001
    for i in range(len(df4)):
            for j in range(i, len(df4)):
                l2 = LA.norm(AA_image[i][:][:]-AA_image[j][:][:],2)
                w = -l2
                #w = 1/eps if l2==0 else 1/l2
                G.add_edge(i,j,weight = w)
                #G.add_edge(i,j,weight = w if w >= cutoff else 0)
    elapsed = time.time() - t
    print(elapsed)
    
    return G, position


def get_nodes_edges_exp(df4):
    t = time.time()
    """
    function to generate nodes and edges
    =input= df4 has to have three columns
    La, Long, AA, and also, AA has to be 200x5 matrix
    After df4.AA = [aacode_to_mat(aacode) for aacode in df4.AA]
    """
    x_y_ = df4[['Longitude','Latitude']].values
    n_nodes = np.arange(len(df4))
    position = dict(zip(n_nodes, x_y_)) 

    AA_mat = df4.iloc[:,-1:]
    AA_image = np.zeros((len(AA_mat), 200, 5))

    for i in range(len(AA_image)):
        AA_image[i][:][:] = AA_mat.iloc[i,0]

    G = nx.Graph()
    cutoff = 1
    eps = 0.001
    for i in range(len(df4)):
            for j in range(i, len(df4)):
                l2 = LA.norm(AA_image[i][:][:]-AA_image[j][:][:],2)
                w = np.exp(-l2)
                #w = np.exp(-l2**2/0.001**2)
                #w = 1/eps if l2==0 else 1/l2
                G.add_edge(i,j,weight = w)
                #G.add_edge(i,j,weight = w if w >= cutoff else 0)
    elapsed = time.time() - t
    print(elapsed)
    
    return G, position


def plot_graph(G, position):
    t = time.time()

    elist = [(u,v) for (u,v,d) in G.edges(data=True)]

    wset=[]        
    for (u,v,d) in G.edges(data=True):
        wset.append(d['weight'])
    
    plt.figure(figsize=(30, 30))
    nx.draw_networkx_nodes(G,position,node_size=1000, node_color = 'w',edgecolors = 'black' )
    nx.draw_networkx_labels(G,position)
    nx.draw_networkx_edges(G,position, edgelist = G.edges, width =  wset)
    plt.axis('off')
    plt.show()


    elapsed = time.time() - t
    print(elapsed)
    
def plot_graph_cut_weighted_edges(G, position, cutoff):
    t = time.time()

    elist = [(u,v) for (u,v,d) in G.edges(data=True)]

    wset=[]        
    for (u,v,d) in G.edges(data=True):
        d = d['weight']
        wset.append(-d+cutoff if d<cutoff else 0)
    
    plt.figure(figsize=(30, 30))
    nx.draw_networkx_nodes(G,position,node_size=1000, node_color = 'w',edgecolors = 'black' )
    #nx.draw_networkx_labels(G,position)
    nx.draw_networkx_edges(G,position, edgelist = G.edges, width = wset, edge_color = 'b')
    plt.axis('on')
    plt.show()


    elapsed = time.time() - t
    print(elapsed)
    
def plot_graph_cut_colored_edges(G, position, cutoff):
    t = time.time()

    elist = [(u,v) for (u,v,d) in G.edges(data=True)]

    wset=[]        
    for (u,v,d) in G.edges(data=True):
        d = d['weight']
        wset.append(-d+cutoff if d<cutoff else 0)
    wset_max = max(wset)
    wset = wset/wset_max *100
    
    plt.figure(figsize=(30, 30))
    #nx.draw(G, position, node_color='black', edgelist=elist, edge_color=wset, width=10.0, edge_cmap=plt.cm.Blues)
    
    nx.draw_networkx_nodes(G,position,node_size=1000, node_color = 'w', edgecolors = 'black')
    #nx.draw_networkx_labels(G,position)
    nx.draw_networkx_edges(G,position, edgelist = G.edges, width =10.0, edge_color = wset, edge_cmap=plt.cm.Blues )
    plt.axis('on')
    plt.show()
    
    elapsed = time.time() - t
    print(elapsed)
    
    
def plot_graph_cut_weighted_edges_exp(G, position, cutoff):
    t = time.time()

    #elist = [(u,v) for (u,v,d) in G.edges(data=True)]

    wset=[] 
    for (u,v,d) in G.edges(data=True):
        dv = d['weight']
        wset.append(dv if dv > cutoff else 0)
                                   
    
    plt.figure(figsize=(30, 30))
    nx.draw_networkx_nodes(G,position,node_size=1000, node_color = 'w',edgecolors = 'black' )
    #nx.draw_networkx_labels(G,position)
    nx.draw_networkx_edges(G,position, edgelist = G.edges, width = wset, edge_color = 'b')
    plt.axis('on')
    plt.show()
    
    
def plot_graph_cut_weighted_edges_exp2(G, position, cutoff, df=None):
    t = time.time()

    #elist = [(u,v) for (u,v,d) in G.edges(data=True)]

    wset=[] 
    for (u,v,d) in G.edges(data=True):
        dv = d['weight']
        wset.append(3 if dv > cutoff else 0)
                                   
    
    list_b = df[df['Farm_Type']=='Breeding Herd'].index.tolist()
    list_n = df[df['Farm_Type']=='Nursery'].index.tolist()
    list_f = df[df['Farm_Type']=='Wean to Finish'].index.tolist()
    #pos_b = {key: value for key, value in position.items() if key in Gb}
    #pos_n = {key: value for key, value in position.items() if key in Gn}
    #pos_f = {key: value for key, value in position.items() if key in Gf}
    
    #plt.figure(figsize=(30, 30))
    
    nx.draw_networkx_nodes(G,position,node_size=1000, nodelist = list_b, node_color = 'yellow',edgecolors = 'black' )
    nx.draw_networkx_nodes(G,position,node_size=1000, nodelist = list_n, node_color = 'green',edgecolors = 'black' )
    nx.draw_networkx_nodes(G,position,node_size=1000, nodelist = list_f, node_color = 'red',edgecolors = 'black' )
    #nx.draw_networkx_labels(G,position)
    nx.draw_networkx_edges(G,position, edgelist = G.edges, width = wset, edge_color = 'b')
    plt.axis('on')
    plt.show()
    
   
    
print('All Imported')
print('All Newly Imported')
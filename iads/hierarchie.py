import scipy.cluster.hierarchy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------

def normalisation(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# ---------------------------

def dist_euclidienne_vect(e1, e2):
    return np.sqrt(((e2 - e1)**2).sum())

# ---------------------------

def dist_manhattan_vect(e1, e2):
    return np.abs(e2 - e1).sum()

# ---------------------------

def dist_vect(s, v1, v2):
    if s == "euclidienne":
        return dist_euclidienne_vect(v1, v2)
    elif s == "manhattan":
        return dist_manhattan_vect(v1, v2)

# ---------------------------

def centroide(m):
    if isinstance(m,list):
        return m[0]
    s = np.sum(m, axis=0)
    return s / m.shape[0]

# ---------------------------

def dist_groupes(s, gv1, gv2):
    cv1 = centroide(gv1)
    cv2 = centroide(gv2)
    if s == "euclidienne":
        return dist_euclidienne_vect(cv1, cv2)
    elif s == "manhattan":
        return dist_manhattan_vect(cv1, cv2)

# ---------------------------

def initialise(M):
    dico = dict()
    for i in range(M.shape[0]):
        dico[i] = [M[i]]
    return dico

# ---------------------------

def fusionne(s, dic):
    k = list(dic.keys())
    res = dic
    c1 = 0
    c2 = 0
    d = 10000000000
    for i in dic.keys():
        for j in dic.keys():
            if i == j:
                continue
            db = dist_groupes(s, dic[i], dic[j])
            if db < d:
                c1 = i
                c2 = j
                d = db
    print("Fusion de ",c1," et ",c2,"pour une distance de ",d)
    m = max(dic.keys())+1
    l = np.concatenate([res[c1], res[c2]])
    del res[c1]
    del res[c2]
    res[m] = l
    return (res, c1, c2, d)

# ---------------------------

def clustering_hierarchique(train, s):
    courant = initialise(train)       # clustering courant, au départ:s données data_2D normalisées
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes à fusionner
        new,k1,k2,dist_min = fusionne(s,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_min,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_min,2] ])
        courant = new
        
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)    
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme à partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(
        M_Fusion,
        leaf_font_size=18.,  # taille des caractères de l'axe des X
    )
    plt.show()
    return M_Fusion

# ---------------------------

def dist_max_groupes(s, c1, c2):
    res = []
    if s == "euclidienne":
        for e in c1:
            for f in c2:
                if np.array_equal(e,f):
                    continue
                res.append(dist_euclidienne_vect(e,f))
        return max(res)
    elif s == "manhattan":
        for e in c1:
            for f in c2:
                if np.array_equal(e,f):
                    continue
                res.append(dist_manhattan_vect(e,f))
        return max(res)

# ---------------------------

def fusionne_max(s, dic):
    k = dic.keys()
    res = dic
    c1 = 0
    c2 = 0
    d = 0
    for i in k:
        for j in k:
            if i == j:
                continue
            db = dist_max_groupes(s, dic[i], dic[j])
            if db > d:
                c1 = i
                c2 = j
                d = db
    print("Fusion de ",c1," et ",c2,"pour une distance de ",d)
    m = max(k)+1
    l = np.concatenate([res[c1], res[c2]])
    del res[c1]
    del res[c2]
    res[m] = l
    return (res, c1, c2, d)

# ---------------------------

def clustering_hierarchique_max(train, s):
    courant = initialise(train)       # clustering courant, au départ:s données data_2D normalisées
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes à fusionner
        new,k1,k2,dist_max = fusionne_max(s,courant)
        if(len(M_Fusion)==0):
            M_Fusion = [k1,k2,dist_max,2]
        else:
            M_Fusion = np.vstack( [M_Fusion,[k1,k2,dist_max,2] ])
        courant = new
        
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)    
    plt.xlabel('Exemple', fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme à partir de la matrice M_Fusion:
    scipy.cluster.hierarchy.dendrogram(
        M_Fusion,
        leaf_font_size=18.,  # taille des caractères de l'axe des X
    )
    plt.show()
    return M_Fusion

# ---------------------------

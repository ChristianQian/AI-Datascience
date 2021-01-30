# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(DF):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    return (DF - DF.min(axis=0)) / (DF.max(axis=0) - DF.min(axis=0))

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.sqrt(((v2 - v1)**2).sum())

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(DF):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return (DF.sum() / DF.shape[0]).to_frame().T

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(DF):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    c = centroide(DF)
    res = 0
    for e in DF.values:
        res += (dist_vect(e, c)**2)
    return res.sum()


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,DF):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    return DF.sample(n=K, axis=0)


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(Exe,Centres):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    res = []
    Centres.apply(lambda x : res.append(dist_vect(x,Exe)), axis=1)
    return res.index(min(res))

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(Base,Centres):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    dico = dict()
    for i in range(len(Centres.index)):
        dico[i] = []
    for t in range(len(Base.index)):
        pp = plus_proche(Base.iloc[t], Centres)
        dico[pp].append(t)
    return dico

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(Base,U):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    l = []
    for k in U.keys():
        cd = Base.iloc[U[k], :]
        moys = cd.mean()
        l.append(moys)
    return pd.DataFrame(l, columns = list(Base))

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(Base, U):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    res = 0
    for k in U.keys():
        cd = Base.iloc[U[k], :]
        res += inertie_cluster(cd)
    return res

# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(K, Base, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    p = initialisation(K, Base)
    m = affecte_cluster(Base, p)
    p = nouveaux_centroides(Base, m)
    j = inertie_globale(Base,m)
    for i in range(iter_max):
        p2 = initialisation(K, Base)
        m2 = affecte_cluster(Base, p2)
        p2 = nouveaux_centroides(Base, m2)
        j2 = inertie_globale(Base, m2)
        dif = np.abs(j2 - j)
        print("iteration ", i+1 ," Inertie : ", j2, " Difference:", dif)
        if dif < epsilon:
            break
        p = p2
        m = m2
        j = j2
    return p2, m2

# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    colors = cm.rainbow(np.linspace(0, 1, len(Affect.keys())+1))
    #colors = itertools.cycle(["b", "c", "g"])
    for i in Affect.keys():
        x = []
        y = []
        tab = Affect[i]
        for e in tab:
            stock = Base.iloc[e]
            x.append(stock['X'])
            y.append(stock['Y'])
        plt.scatter(x, y, color=colors[i])
    plt.scatter(Centres['X'],Centres['Y'],color=colors[len(Affect.keys())],marker='x')
    plt.show()
# -------

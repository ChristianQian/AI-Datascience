# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de 3i026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from . import Classifiers as cl
import random

# importation de LabeledSet
from . import LabeledSet as ls

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1

def plot_frontiere(set,classifier,step=10):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])
    
# ------------------------ 

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ 
        rend un LabeledSet 2D généré aléatoirement.
        Arguments:
        - positive_center (vecteur taille 2): centre de la gaussienne des points positifs
        - positive_sigma (matrice 2*2): variance de la gaussienne des points positifs
        - negative_center (vecteur taille 2): centre de la gaussienne des points négative
        - negative_sigma (matrice 2*2): variance de la gaussienne des points négative
        - nb_points (int):  nombre de points de chaque classe à générer
    """

    lset = ls.LabeledSet(2)
    pos = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    neg = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    
    for e in pos:
        lset.addExample(e, 1)
    for f in neg:
        lset.addExample(f, -1)
    return lset
    
# ------------------------ 

class KernelBias:
    def transform(self,x):
        y=np.asarray([x[0],x[1],1])
        return y

# ------------------------ 

class KernelPoly:
    def transform(self,x):
        y=np.asarray([1, x[0],x[1], x[0]**2, x[1]**2, x[0]*x[1]])
        return y

# ------------------------ 

def split(labeledSet, p):
    d = labeledSet.getInputDimension()
    l1 = ls.LabeledSet(d)
    l2 = ls.LabeledSet(d)
    taille = labeledSet.size()
    
    for i in range(int(taille * p)):
        l1.addExample(labeledSet.getX(i), labeledSet.getY(i))
        
    for j in range(int(taille * p), taille):
        l2.addExample(labeledSet.getX(j), labeledSet.getY(j))
        
    return (l1,l2)

# ------------------------ 

def affiche_base(LS):
    """ LabeledSet
        affiche le contenu de LS
    """
    for i in range(0,LS.size()):
        print("Exemple "+str(i))
        print("\tdescription : ",LS.getX(i))
        print("\tlabel : ",LS.getY(i))
    return

# ------------------------

def classe_majoritaire(labeledset):
    nbr = len(labeledset.x[np.where(labeledset.y == 1),:][0])
    if nbr >= (labeledset.size() / 2) :
        return 1
    return -1 

# ------------------------ 

def shannon(p):
    k = len(p)
    h = 0
    for e in p:
        if e == 0:
            continue
        h -= e * math.log(e, k)
    return h

# ------------------------ 

def entropie(labeledset):
    nbr1 = len(labeledset.x[np.where(labeledset.y == 1),:][0]) / labeledset.size()
    nbr2 = 1 - nbr1
    return shannon([nbr1, nbr2])

# ------------------------ 

def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)

# ------------------------ 

def divise(LSet,att,seuil):
    d = LSet.getInputDimension()
    set1 = ls.LabeledSet(d)
    set2 = ls.LabeledSet(d)
    taille = LSet.size()
    
    for i in range(taille):
        x = LSet.getX(i)
        y = LSet.getY(i)
        if(x[att] <= seuil):
            set1.addExample(x,y)
        else:
            set2.addExample(x,y)
    return (set1,set2)

# ------------------------ 

def construit_AD(LSet,epsilon):
    un_arbre= cl.ArbreBinaire()
    e = entropie(LSet)
    if e <= epsilon:
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
    else:
        ss = []
        es = []
        for c in range(LSet.getInputDimension()):
            a, b = discretise(LSet, c)
            ss.append(a)
            es.append(b)
        iatt = es.index(min(es))
        Linf, Lsup = divise(LSet,iatt,ss[iatt])
        un_arbre.ajoute_fils(construit_AD(Linf,epsilon),construit_AD(Lsup,epsilon),iatt,ss[iatt])
    return un_arbre

# ------------------------ 

def construit_ad(LSet,epsilon):
    un_arbre= cl.ArbreBinaire()
    e = entropie(LSet)
    if e <= epsilon:
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
    else:
        ss = []
        es = []
        for c in range(LSet.getInputDimension()):
            a, b = discretise(LSet, c)
            ss.append(a)
            es.append(b)
        iatt = es.index(min(es))
        if e - es[iatt] <= epsilon:
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))
            print("test")
        else:
            Linf, Lsup = divise(LSet,iatt,ss[iatt])
            un_arbre.ajoute_fils(construit_AD(Linf,epsilon),construit_AD(Lsup,epsilon),iatt,ss[iatt])
    return un_arbre

# ------------------------ 

def tirage(VX, m, r):
    if r:
        res = []
        for _ in range(m):
            res.append(random.choice(VX))
        return res
    return random.sample(VX, m)

# ------------------------ 

def echantillonLS(X, m, r):
    LS = ls.LabeledSet(X.getInputDimension())
    VX = list(range(X.size()))
    l = tirage(VX, m, r)
    for i in l:
        LS.addExample(X.getX(i), X.getY(i))
    return LS

# ------------------------ 

def construit_AD_aleatoire(LSet,epsilon,nbatt):
    un_arbre= cl.ArbreBinaire()
    e = entropie(LSet)
    if e <= epsilon:
        un_arbre.ajoute_feuille(classe_majoritaire(LSet))
    else:
        ss = []
        es = []
        #for c in range(nbatt):
        for _ in range(nbatt):
            c = random.randint(0, LSet.getInputDimension()-1)
            a, b = discretise(LSet, c)
            ss.append(a)
            es.append(b)
        iatt = es.index(min(es))
        if e - es[iatt] <= epsilon:
            un_arbre.ajoute_feuille(classe_majoritaire(LSet))
        else:
            Linf, Lsup = divise(LSet,iatt,ss[iatt])
            un_arbre.ajoute_fils(construit_AD_aleatoire(Linf,epsilon,nbatt),construit_AD_aleatoire(Lsup,epsilon,nbatt),iatt,ss[iatt])
    return un_arbre

# ------------------------ 

# ------------------------ 

# ------------------------  
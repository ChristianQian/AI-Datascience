# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import random
import graphviz as gv
from . import LabeledSet as ls
from . import utils as ut

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        acc = 0
        for i in range(dataset.size()):
            if self.predict(dataset.getX(i)) == dataset.getY(i):
                acc += 1
        return (acc / dataset.size())
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension           
        self.w = np.array([np.random.randint(1, 3, input_dimension)])
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        if np.vdot(x,self.w) < 0:
            return -1
        return 1

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        raise NotImplementedError("Please Implement this method")
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        l = []
        for i in range(self.labeledSet.size()):
            l.append(np.linalg.norm(self.labeledSet.getX(i)-x))
        ind = np.argsort(l)
        res = []
        for j in range(self.k):
            res.append(self.labeledSet.getY([ind[j]]))
        s = sum(res)
        if s < 0:
            return -1
        return 1

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        self.labeledSet = labeledSet

# ---------------------------

class ClassifierPerceptronRandom(Classifier):
    def __init__(self, input_dimension):
        """ Argument:
                - input_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = (2* v - 1) / np.linalg.norm(v) # on normalise par la norme de v

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z
        
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("No training needed")

# ---------------------------

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        v = np.random.rand(input_dimension)
        self.w = (2* v - 1) / np.linalg.norm(v)

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        if np.dot(x, self.w) >= 0:
            return 1
        return -1
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet = labeledSet
        r = list(range(0,labeledSet.size(),1))
        random.shuffle(r)
        for i in range(labeledSet.size()):
            x = labeledSet.getX(r[i])
            self.w += self.learning_rate * (labeledSet.getY(r[i])-self.predict(x)) * x
        return self.w

# ---------------------------

class ClassifierPerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dimension_kernel = dimension_kernel
        self.learning_rate = learning_rate
        self.kernel = kernel
        v = np.random.rand(dimension_kernel)
        self.w = (2* v - 1) / np.linalg.norm(v)
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        kx = self.kernel.transform(x)
        if np.dot(kx, self.w) >= 0:
            return 1
        return -1
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        r = list(range(0,labeledSet.size(),1))
        random.shuffle(r)
        for i in range(labeledSet.size()):
            x = labeledSet.getX(r[i])
            kx = self.kernel.transform(x)
            self.w += self.learning_rate * kx * (labeledSet.getY(r[i])-self.predict(x))
        return self.w

# ---------------------------

class ClassifierGradiantStochastique(Classifier):
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        v = np.random.rand(input_dimension)
        self.w = (2* v - 1) / np.linalg.norm(v)
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(x, self.w)
      
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        r = list(range(0,labeledSet.size(),1))
        random.shuffle(r)
        for i in range(labeledSet.size()):
            x = labeledSet.getX(r[i])
            self.w += self.learning_rate * (labeledSet.getY(r[i])-np.dot(x, self.w)) * x
        return self.w
    
    def loss(self, dataset):
        l = 0
        for i in range(dataset.size()):
            l += (dataset.getY(i) - np.dot(self.w, dataset.getX(i)))**2
        return l / dataset.size()

# ---------------------------

class ClassifierGradiantBatch(Classifier):
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        v = np.random.rand(input_dimension)
        self.w = (2* v - 1) / np.linalg.norm(v)
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(x, self.w)

    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        r = list(range(0,labeledSet.size(),1))
        random.shuffle(r)
        g = 0
        for i in range(labeledSet.size()):
            x = labeledSet.getX(r[i])
            g += (labeledSet.getY(r[i])-np.dot(self.w, x)) * x
        self.w += self.learning_rate * g
        return self.w
    
    def loss(self, dataset):
        l = 0
        for i in range(dataset.size()):
            l += (dataset.getY(i) - np.dot(self.w, dataset.getX(i)))**2
        return l / dataset.size()

# ---------------------------

class ClassifierGradiantStochastiqueKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dimension_kernel = dimension_kernel
        self.learning_rate = learning_rate
        self.kernel = kernel
        v = np.random.rand(dimension_kernel)
        self.w = (2* v - 1) / np.linalg.norm(v)
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        kx = self.kernel.transform(x)
        if np.dot(kx, self.w) >= 0:
            return 1
        return -1
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        r = list(range(0,labeledSet.size(),1))
        random.shuffle(r)
        for i in range(labeledSet.size()):
            x = labeledSet.getX(r[i])
            kx = self.kernel.transform(x)
            self.w += self.learning_rate * (labeledSet.getY(r[i])-np.dot(kx, self.w)) * kx
        return self.w
    
    def loss(self, dataset):
        l = 0
        for i in range(dataset.size()):
            x = dataset.getX(i)
            kx = self.kernel.transform(x)
            l += (dataset.getY(i) - np.dot(self.w, kx))**2
        return l / dataset.size()

# ---------------------------

class ClassifierGradiantBatchKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dimension_kernel = dimension_kernel
        self.learning_rate = learning_rate
        self.kernel = kernel
        v = np.random.rand(dimension_kernel)
        self.w = (2* v - 1) / np.linalg.norm(v)
        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        kx = self.kernel.transform(x)
        if np.dot(kx, self.w) >= 0:
            return 1
        return -1
    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        r = list(range(0,labeledSet.size(),1))
        random.shuffle(r)
        g = 0
        for i in range(labeledSet.size()):
            x = labeledSet.getX(r[i])
            kx = self.kernel.transform(x)
            g += (labeledSet.getY(r[i])-np.dot(kx, self.w)) * kx
        self.w += self.learning_rate * g
        return self.w
    
    def loss(self, dataset):
        l = 0
        for i in range(dataset.size()):
            x = dataset.getX(i)
            kx = self.kernel.transform(x)
            l += (dataset.getY(i) - np.dot(self.w, kx))**2
        return l / dataset.size()

# ---------------------------

class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g

# ---------------------------

class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = ut.construit_AD(set,self.epsilon)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
        
# ---------------------------

class ClassifierBaggingTree(Classifier):
    def __init__(self, B, p, seuil, r):
        self.B = B
        self.p = p
        self.r = r
        self.seuil = seuil
        self.foret = set()
        
    def predict(self,x):
        cpt = 0
        for a in self.foret:
            cpt += a.predict(x)
        if cpt >= 0:
            return 1
        return -1
    
    def train(self, labeledSet):
        for _ in range(self.B):
            e = ut.echantillonLS(labeledSet, int(self.p * labeledSet.size()), self.r)
            ad = ArbreDecision(self.seuil)
            ad.train(e)
            self.foret.add(ad)
        return self.foret

# ---------------------------

class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    def __init__(self, B, p, seuil, r):
        super(ClassifierBaggingTreeOOB, self).__init__(B, p, seuil, r)
        
    def taux(self):
        if self.B == 0:
            return 0
        sglobal = 0
        for adi,ti in self.foret:
            if ti.size() == 0:
                continue
            t= 0
            for j in range(ti.size()):
                if adi.predict(ti.getX(j)) == ti.getY(j):
                    t += 1
            sglobal += t/ti.size()

        return sglobal / self.B

        
    def train(self, labeledSet):
        for _ in range(self.B):
            index = ut.tirage(list(range(labeledSet.size())) , int(self.p * labeledSet.size()) , self.r)
            xi = ls.LabeledSet(labeledSet.getInputDimension())
            ti = ls.LabeledSet(labeledSet.getInputDimension())
            for j in index:
                xi.addExample(labeledSet.getX(j), labeledSet.getY(j))

            for j in range(labeledSet.size()):
                if j not in index:
                    ti.addExample(labeledSet.getX(j), labeledSet.getY(j))
            ad = ArbreDecision(self.seuil)
            ad.train(xi)
            self.foret.add((ad,ti))
        return xi, ti

# ---------------------------

class ArbreDecisionAleatoire(ArbreDecision):
    def __init__(self, epsilon, nbatt):
        self.nbatt = nbatt
        super(ArbreDecisionAleatoire, self).__init__(epsilon)
        
    def train(self,set): 
        self.set=set
        self.racine = ut.construit_AD_aleatoire(set,self.epsilon,self.nbatt)

# ---------------------------

class ClassifierRandomForest(ClassifierBaggingTree):
    def __init__(self, B, p, seuil, r):
        super(ClassifierRandomForest, self).__init__(B, p, seuil, r)
        
    def train(self, labeledSet):
        # respred = []
        for _ in range(self.B):
            # pos = 0
            # neg = 0
            dim = labeledSet.getInputDimension()
            index = ut.tirage(list(range(labeledSet.size())) , int(self.p * labeledSet.size()) , self.r)
            xi = ls.LabeledSet(dim)
            for i in index:
                xi.addExample(labeledSet.getX(i), labeledSet.getY(i))
            nbatt = random.randint(1,dim)
            ad = ArbreDecisionAleatoire(self.seuil, nbatt)
            ad.train(xi)
            self.foret.add(ad)
            return self.foret

# ---------------------------

# ---------------------------
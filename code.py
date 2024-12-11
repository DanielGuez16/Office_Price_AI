import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn. linear_model import LinearRegression

#Lecture des données d'entrée
def read_input():
    #Lire F (nombre de caractéristiques/colonnes) et N (nombre de lignes d'entrainement)
    F, N = map(int, input().strip().split())
    #input reçoit une ligne de texte et retourne un string 2 6 = "2 6"
    #.strip() appliquée au string de input et supprime les \n, \t etc...
    #.split() divise un string en plusieurs avec par défaut ' ' "2 6" = ["2", "6"]
    #map() applique une fonction (ici int) à chaque élément d'un itérable
    #["2", "6"] = [2, 6] CELA RESTE UN ITERABLE PAS UNE LISTE
    
    #Lire les données d'entrainement
    train_data = []
    for _ in range(N):
        train_data.append(list(map(float, input().strip().split())))
        #list() convertie un objet itérable en une liste [] permet de ne pas avoir un itérable ici c'est une vraie liste pas seulement des valeurs
    train_data = np.array(train_data)
    #on a donc le premier tableau de N lignes et F+1 colonnes
    
    #Lire T (nombre de lignes de test)
    T = int(input().strip())
    #pas besoin du split c'est juste une valeur
    
    #Lire les données de test
    test_data = []
    for _ in range(T):
        test_data.append(list(map(float, input().strip().split())))
    test_data = np.array(test_data)
    
    return F, train_data, T, test_data

#Prédiction des prix par régression polynomiale    
def predict_prices(F, train_data, test_data):
    #Séparer les caractéristiques (X) et les prix (y) dans les données d'entrainement
    X_train = train_data[:, :F] #je veux toutes les lignes :  , je veux toutes les colonnes de 0 à F-1 :F.
    y_train = train_data[:, F]#je veux toutes les lignes :  , je veux Fème colonne.
    
    #Initialiser la transformation polynomiale (ordre 3 maximum)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_train) #fit permet d'apprendre PRIMORDIAL
    #voir documentation sur PolynomialFeatures sur fit_transform dans scikit-learn
    
    #Modèle de régression linéaire (appliqué aux caractéristiques polynomiales)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    #Transformer les données de test
    X_test_poly = poly.transform(test_data) #Surtout PAS de FIT sinon "fuite de données" on apprend des données de test
    
    #Prédire les prix pour les données de test grace à predict en se basant sur model.fit qui a appris
    predictions = model.predict(X_test_poly)
    return predictions
    
#Fonction principale
def main():
    F, train_data, T, test_data = read_input()
    predictions = predict_prices(F, train_data, test_data)
    
    #Afficher les prédictions pour chaque ligne de test
    for price in predictions:
        print(price)
        
# Executer le programme

if __name__ == "__main__":
    main()
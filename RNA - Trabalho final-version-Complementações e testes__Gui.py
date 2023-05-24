# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.initializers import RandomNormal, Constant
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

#Verificando o diretório
# os.getcwd()
os.chdir("C:\\Users\\Treinamento\\Documents\\RNA")
#os.getcwd()

#Importando os arquivos
df = pd.read_csv('arquivo.csv')
df = df.drop(df.columns[0], axis=1)
df = df.drop(columns=['E2'], axis=1)
previsores = df.iloc[:, 0:15].values
classe = df.iloc[:, 16].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
df.drop_duplicates(inplace=True)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, 
classe_dummy, 
shuffle=True,
random_state = 10,
test_size=0.10)

# previsores_treinamento = previsores
# classe_treinamento = classe_dummy

def criarRede():
    # Modelo
    model = Sequential([
        Dense(512, input_dim = 15, activation="relu"),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(400, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(4, activation='sigmoid')
    ])
 
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model
model = criarRede()

history = model.fit(
    previsores_treinamento, 
    classe_treinamento, 
    batch_size = 8000, epochs = 10,
    validation_split=0.25,
    use_multiprocessing= True
)


model.summary()

%matplotlib inline
%config InlineBackend.figure_format = 'svg'
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history, 'accuracy')

plot_metric(history, 'loss')

#Validado o modelo
model.evaluate(previsores_teste, classe_teste, verbose=2)


#Exibindo a matriz de confusão
previsoes = model.predict(previsores_teste)
previsoes = (previsoes > 0.5)
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
matriz = confusion_matrix(previsoes2, classe_teste2)
labels = ["Classe 0", "Classe 1"]
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#Salvando o modelo gerado
model_json = model.to_json()
with open('rna-trabalhofinal-version.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('rna-trabalhofinal-version.h5')
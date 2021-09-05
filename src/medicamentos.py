import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# estilo de visualizacao dos dados plotados
plt.style.use('ggplot') 


# diretorio base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#print('BASE_DIR', BASE_DIR)

# join para BASE_DIR com diretorio filho
DATA_DIR = os.path.join(BASE_DIR, 'data')
#print('DATA_DIR', DATA_DIR)

# list compreensions do dataset
file_names = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]
#print('FILE_NAME', file_names)

# df = dataframe - file_names
for i in file_names:
    df = pd.read_csv(os.path.join(DATA_DIR, i))

# tratamento dos dados e mapeamento
# Remanejamento = 1
# Entrega Realizada = 2
# Remanejamento para Malaria = 3
map_data = {'R': 1, 'ER': 2, 'M': 3}
df['STATUS'] = df['STATUS'].map(map_data)
##print('Alteracao de valores categóricos: \n', df.head(50))

map_data_region = {'NORTE': 1, 'NORDESTE': 2, 'CENTRO-OESTE': 3, 'SUDESTE': 4, 'SUL': 5}
df['REGIAO'] = df['REGIAO'].map(map_data_region)
##print('Alteracao de valores categóricos: \n', df.head(50))

map_data_progsaude = {'COVID-19': 1, 'INFLUENZA': 2}
df['PROGSAUDE'] = df['PROGSAUDE'].map(map_data_progsaude)
##print('Alteracao de valores categóricos: \n', df.head(50))

# num e pandas
def ver_amostras_das_classes_status():
    sample_1 = np.where(df.loc[df['STATUS'] == 1])
    sample_2 = np.where(df.loc[df['STATUS'] == 2])
    sample_3 = np.where(df.loc[df['STATUS'] == 3])
    print('\nAmostra da classe 1 - Remanejamento: ', sample_1)
    print('\nAmostra da classe 2 - Entrega realizada: ', sample_2)
    print('\nAmostra da classe 3 - Remanejamento Malaria: ', sample_3)

# num e pandas
def ver_amostras_das_classes_regiao():
    sample_1 = np.where(df.loc[df['REGIAO'] == 1])
    sample_2 = np.where(df.loc[df['REGIAO'] == 2])
    sample_3 = np.where(df.loc[df['REGIAO'] == 3])
    sample_4 = np.where(df.loc[df['REGIAO'] == 4])
    sample_5 = np.where(df.loc[df['REGIAO'] == 5])
    print('\nAmostra da classe 1 - Norte: ', sample_1)
    print('\nAmostra da classe 2 - Nordeste: ', sample_2)
    print('\nAmostra da classe 3 - Centro-Oeste: ', sample_3)
    print('\nAmostra da classe 4 - Sudeste: ', sample_4)
    print('\nAmostra da classe 5 - Sul: ', sample_5)


# quantidade de amostras por classe
def ver_qtde_amostras_por_classe_status():
    vl_remanejamento = len(df.loc[df['STATUS'] == 1])
    vl_entrega_realizada = len(df.loc[df['STATUS'] == 2])
    vl_remanejamento_malaria = len(df.loc[df['STATUS'] == 3])
    print('\nAmostra da classe 1 - Remanejamento: ', vl_remanejamento)
    print('\nAmostra da classe 2 - Entrega Realizada: ', vl_entrega_realizada)
    print('\nAmostra da classe 3 - Remanejamento Malaria: ', vl_remanejamento_malaria)

def ver_qtde_amostras_por_classe_regiao():
    vl_norte = len(df.loc[df['REGIAO'] == 1])
    vl_nordeste = len(df.loc[df['REGIAO'] == 2])
    vl_centro_oeste = len(df.loc[df['REGIAO'] == 3])
    vl_sudeste = len(df.loc[df['REGIAO'] == 4])
    vl_sul = len(df.loc[df['REGIAO'] == 5])
    print('\nAmostra da classe 1 - Norte: ', vl_norte)
    print('\nAmostra da classe 2 - Nordeste: ', vl_nordeste)
    print('\nAmostra da classe 3 - Centro-Oeste: ', vl_centro_oeste)
    print('\nAmostra da classe 4 - Sudeste: ', vl_sudeste)
    print('\nAmostra da classe 5 - Sul: ', vl_sul)


# verifica se existe dados faltantes no conjunto de dados

dt_feature = df.iloc[:, : -1]
dt_target = df.iloc[:, -1]
dt_feature = dt_feature.mask(dt_feature == 1).fillna(dt_feature.mean)
##print('DT_FEATURE: ', dt_feature)
##print('DT_TARGET: ', dt_target)


# plotando os dados - funcoes
def plot_hist():
    #histograma de classes
    plt.hist(df.iloc[:,1], color='b', width=.1)
    plt.xlabel('Qtde. Amostras')
    plt.ylabel('Hist da Classe')
    plt.show()

# histograma web offline [verificar erro na execucao]
def target_count():
    trace = go.Bar(x = df['STATUS'].value_counts().values.tolist(),
                y = ['saudaveis', 'diabeticos'],
                orientation = 'v',
                text = df['STATUS'].value_counts().values.tolist(),
                textfont = dict(size=15),
                textposition = 'auto',
                opacity = 0.8, marker=dict(color=['lightskyblue', 'gold'],
                line=dict(color='#000000', width=1.5)))
    layout = dict(title='resultado')
    fig = dict(data=[trace], layout = layout)
    py.iplot(fig)


# analise de correlacao
def correlation(size=5):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


# bloxsplot
def bloxplot():
    f, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor('#fafafa')
    ax.set(xlim=(-0.5, 200))
    plt.ylabel('quantidade')
    plt.title('Overview Dataset')
    ax = sns.boxplot(data=df, orient='v', palette='Set2')
    plt.show()



# lista de armazenamento de acuracia
accuracy_PC = []

# vetor beisiano Naive Bayes
accuracy_NB = []

def split_model():

    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size=0.3, random_state=1)
        print('divisao do conjunto de dados\n')
        print('x_train: %d\n y_train %d\n x_test %d\n y_test %d\n' %(len(x_train), len(y_train), len(x_test), len(y_test)))
        print('quantidade de amostras da classe 1: ', len(y_train.loc[y_train == 1]))
        print('quantidade de amostras da classe 2: ', len(y_train.loc[y_train == 2]))
        print('quantidade de amostras da classe 3: ', len(y_train.loc[y_train == 3]))

        # Perceptron
        #percep = Perceptron(random_state=i)
        ##percep = Perceptron()
        ##percep.fit(x_train, y_train) #treinar em cima do conjunto de treinamento
        ##percep.predictions = percep.predict(x_test) # testar pra mim
        ##acc_percep = percep.score(x_test, y_test) # apresentar o resultado

        # Naive Bayes
        ##gnb = GaussianNB() #criado o classificador
        ##gnb.fit(x_train, y_train) # treinar o classificador
        ##gnb.predictions = gnb.predict(x_test) #testar o classificador com o conjunto de test
        ##acc_nb = gnb.score(x_test, y_test) # apresentar o resultado

        # Accuracy
        ##accuracy_PC.append(acc_percep)
        ##accuracy_NB.append(acc_nb)

        ##print('\n Resultados Perceptron: \n Acc_Perceptron: ', acc_percep)
        ##print('\n Resultados NB: \n Acc_Perceptron: ', acc_nb)
        ##print(metrics.confusion_matrix(y_test, percep.predictions))
        ##print('\n Classificacao: \n', metrics.classification_report(y_test, percep.predictions))

        ##print('\n Vetor de accuracia Peceptron: ', accuracy_PC)
        ##print('\n Vetor de accuracia NB: ', accuracy_NB)

        ##median = np.mean(accuracy_PC)
        ##print('Vetor accuracy_PC - Media: ', median)



# chamada das funcoes

##ver_amostras_das_classes_status()
##ver_amostras_das_classes_regiao()
##ver_qtde_amostras_por_classe_status()
##ver_qtde_amostras_por_classe_regiao()
##ver_dados_faltantes()

##plot_hist()
##target_count()
##correlation()
##bloxplot()
split_model()



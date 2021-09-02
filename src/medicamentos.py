import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py


# estilo de visualizacao dos dados plotados
plt.style.use('ggplot') 


# diretorio base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# join para BASE_DIR com diretorio filho
DATA_DIR = os.path.join(BASE_DIR, 'data')

# list compreensions do dataset
file_names = [i for i in os.listdir(DATA_DIR) if i.endswith('.csv')]

# df = dataframe - file_names
for i in file_names:
    df = pd.read_csv(os.path.join(DATA_DIR, i))



# plotando os dados - funcoes
def plot_hist():
    # histograma de classes
    plt.hist(df.iloc[:,-1], color='b', width=.1)
    plt.xlabel('Qtd Amostra')
    plt.ylabel('Hist da Classe')
    plt.show()

# histograma web offline [verificar erro na execucao]
def target_count():
    trace = go.Bar(x = df['diabetes'].value_counts().values.tolist(),
                y = ['saudaveis', 'diabeticos'],
                orientation = 'v',
                text = df['diabetes'].value_counts().values.tolist(),
                textfont = dict(size=15),
                textposition = 'auto',
                opacity = 0.8, marker=dict(color=['lightskyblue', 'gold'],
                line=dict(color='#000000', width=1.5)))
    layout = dict(title='resultado')
    fig = dict(data=[trace], layout = layout)
    py.iplot(fig)




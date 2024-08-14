import matplotlib.pyplot as plt
import numpy as np

# Dados
N = 6  # Número de grupos
ind = np.arange(N)  # As posições das barras agrupadas
largura = 0.2  # A largura das barras
# Alturas das barras agrupadas
#GGNN = [77.8, 54.8, 92.4, 89.0, 93.7, 90.0]
GGNN = [78.4, 55.3, 92.4, 90.2, 93.8, 90.9]
GraphSAGE_Mean = [76.5, 48.7, 88.0, 85.0, 92.1, 87.6]
GraphSAGE_Pooling = [72.1, 29.9, 68.0, 68.6, 70.6, 71.2]
GraphSAGE_Lstm = [77.1, 49.9, 99.1, 92.3, 98.7, 92.0]

# Criar as barras agrupadas
fig, ax = plt.subplots(figsize=(12, 8))  # Tamanho da figura
barras_GGNN = ax.bar(ind, GGNN, largura, label='GGNN')
barras_GraphSAGE_Mean = ax.bar(ind + largura, GraphSAGE_Mean, largura, label='GraphSAGE-Mean')
barras_GraphSAGE_Pooling = ax.bar(ind + 2*largura, GraphSAGE_Pooling, largura, label='GraphSAGE-Pooling')
barras_GraphSAGE_Lstm = ax.bar(ind + 3*largura, GraphSAGE_Lstm, largura, label='GraphSAGE-LSTM')

# Adicionar rótulos
ax.set_xlabel('Rótulos', fontsize=16)  # Aumentando o tamanho da fonte para o rótulo do eixo x
ax.set_ylabel('Acurácia (%)', fontsize=16)  # Aumentando o tamanho da fonte para o rótulo do eixo y
#ax.set_title('Porcentagem da acurácia por Rótulo', fontsize=16)  # Aumentando o tamanho da fonte para o título
ax.set_xticks(ind + 3*largura / 2)
ax.set_xticklabels(('0 (M)', '1 (J1)', '4 (UC1)', '5 (UC2)', '6 (UC3)', '7 (UC4)'), fontsize=16)  # Aumentando o tamanho da fonte para os rótulos do eixo x
ax.legend(fontsize=12)  # Aumentando o tamanho da fonte para a legenda
ax.set_yticks([0, 20, 40, 60, 80, 100])  # Definindo os locais dos ticks no eixo y
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=16)  # Rotulando os ticks do eixo y

# Adicionar rótulos nas barras
def autolabel(barras):
    """Função para adicionar rótulos nas barras."""
    for barra in barras:
        altura = barra.get_height()
        ax.annotate('{}'.format(altura),
                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 3),  # Deslocamento vertical
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)

autolabel(barras_GGNN)
autolabel(barras_GraphSAGE_Mean)
autolabel(barras_GraphSAGE_Pooling)
autolabel(barras_GraphSAGE_Lstm)

plt.show()


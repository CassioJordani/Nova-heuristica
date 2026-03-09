import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits

digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

# HEURÍSTICA ANTERIOR (DE FORMA ALEATÓRIA)
def heuristica_aleatoria():
    return random.randint(0, 9)

# NOVA HEURÍSTICA (AVALIA O PESO LATERAL DOS PIXELS DA IMAGEM)
def heuristica_lados(row):
    # Basicamente ele soma o lado esquerdo e o lado direito da imagem.
    # Se forem equilibrados, a chance de ser 0 ou 8, dígitos que são mais simétrocps.
    # Caso for muito pesado para um lado tem a chance de ser 1, 4 ou 6, por exemplo.
  
    matriz = row.values.reshape(8, 8)
    lado_esquerdo = matriz[:, :4].sum()  # Das primeiras 4 colunas
    lado_direito = matriz[:, 4:].sum()   # Últimas 4 colunas
    
    # Critério simples de equilíbrio
    diferenca = abs(lado_esquerdo - lado_direito)
    
    if diferenca < 15: 
        return 0  # Muito equilibrado, chuta zero
    elif lado_direito > lado_esquerdo:
        return 1  # Mais peso na direita, retorna o número 1
    else:
        return 6  # Mais peso na esquerda, retorna número 6

# EXECUTANDO AS DUAS HEURÍSTICAS AO MESMO TEMPO

def aplicar_modelos(row):
    # Rodando as as duas
    pred_aleatoria = heuristica_aleatoria()
    pred_lados = heuristica_lados(row)
    
    # CRITÉRIO: Se a imagem tiver poucos pixel, 
    # usamos a aleatória. Se tiver bastante, usamos a de lados.
    if row.sum() < 200:
        return pred_aleatoria
    else:
        return pred_lados

# Aplicando e criando as colunas
df['pred_final'] = df.drop('target', axis=1).apply(aplicar_modelos, axis=1)

# MOSTRANDO OS ACERTOS E A % DE ACURÁCIA
acertos = (df['target'] == df['pred_final']).sum()
acuracia = acertos / len(df)

print(f"Total de registros: {len(df)}")
print(f"Acurácia Final: {acuracia:.4f}")

# exemplo
idx = random.randint(0, len(df)-1)
plt.gray()
plt.matshow(digits.images[idx])
plt.title(f"Alvo: {df['target'].iloc[idx]} | Predição: {df['pred_final'].iloc[idx]}")
plt.show()

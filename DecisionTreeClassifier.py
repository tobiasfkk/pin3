import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_size = 0.6
val_size = 0.2
test_size = 0.2

arquivo_csv = 'peras.csv' # caminho do arquivo
dados = pd.read_csv(arquivo_csv) # leituras dos dados
print(dados.head())
dados.info() # verificação se tem dados nulos

fig = px.histogram(dados, x = 'qualidade', text_auto = True)
fig.write_html('histograma_qualidade.html', auto_open=True)
#Histograma mostrou que tem 2004 boa e 1996 ruim 

fig2 = px.box(dados, x = 'tamanho', color = 'qualidade')
fig2.write_html('histograma_qualidade2.html', auto_open=True)

fig3 = px.box(dados, x = 'peso', color = 'qualidade')
fig3.write_html('histograma_qualidade3.html', auto_open=True)

fig4 = px.box(dados, x = 'docura', color = 'qualidade')
fig4.write_html('histograma_qualidade4.html', auto_open=True)

fig5 = px.box(dados, x = 'crocancia', color = 'qualidade')
fig5.write_html('histograma_qualidade5.html', auto_open=True)

fig6 = px.box(dados, x = 'suculencia', color = 'qualidade')
fig6.write_html('histograma_qualidade6.html', auto_open=True)

fig7 = px.box(dados, x = 'maturacao', color = 'qualidade')
fig7.write_html('histograma_qualidade7.html', auto_open=True)

fig8 = px.box(dados, x = 'acidez', color = 'qualidade')
fig8.write_html('histograma_qualidade8.html', auto_open=True)

x = dados.drop('qualidade', axis = 1)
y = dados['qualidade']

print(x.head())

print(y.head())

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

df = pd.DataFrame(y)
print(df.head())

x_trainval, x_test, y_trainval, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=val_size/(val_size + train_size), random_state=42)

depth_range = range(3, 9)
results = []
best_val_accuracy = 0
best_depth = 0

for depth in depth_range:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia na validação com profundidade {depth}: {val_accuracy}")
    
    # Guardar resultados em uma lista
    results.append((depth, val_accuracy))
    
    # Atualizar o melhor modelo com base na acurácia de validação
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_depth = depth

# Usar a melhor profundidade encontrada para treinar o modelo final
final_model = DecisionTreeClassifier(max_depth=best_depth)
final_model.fit(x_trainval, y_trainval)  # Treinar no conjunto de treinamento + validação

# Fazer previsões no conjunto de teste
y_test_pred = final_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Acurácia no teste com melhor profundidade {best_depth}: {test_accuracy}")


# Imprimir todos os resultados
for depth, val_acc in results:
    print(f"Profundidade: {depth}, Acurácia de Validação: {val_acc:.4f}")

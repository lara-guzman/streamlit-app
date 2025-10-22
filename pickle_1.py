# pickle_1.py
# -----------------------------------------------
# Entrena un modelo RandomForest para predecir
# la especie de pingüinos y guarda el modelo + metadatos en archivos .pickle
# -----------------------------------------------

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. Leer dataset
df = pd.read_csv('penguins.csv')
df.dropna(inplace=True)

# 2. Variables objetivo (y) y predictoras (X)
y = df['species']
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
        'body_mass_g', 'island', 'sex']]

# 3. Codificación de variables categóricas
X = pd.get_dummies(X)
y, uniques = pd.factorize(y)

print('🔹 Especies únicas:', list(uniques))
print('🔹 Variables de entrada (tras get_dummies):', list(X.columns))

# 4. Entrenamiento
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)

# 5. Evaluación
y_pred = rfc.predict(x_test)
score = accuracy_score(y_test, y_pred)
print('✅ Precisión del modelo: {:.2f}%'.format(score * 100))

# 6. Guardar modelo, etiquetas y columnas
with open('random_forest_penguin.pickle', 'wb') as f:
    pickle.dump(rfc, f)

with open('output_penguin.pickle', 'wb') as f:
    pickle.dump(uniques, f)

with open('model_columns_penguin.pickle', 'wb') as f:
    pickle.dump(list(X.columns), f)

print('💾 Archivos generados:')
print(' - random_forest_penguin.pickle')
print(' - output_penguin.pickle')
print(' - model_columns_penguin.pickle')

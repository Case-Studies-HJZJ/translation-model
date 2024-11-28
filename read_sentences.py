import pandas as pd

# Ruta al archivo .tsv
file_path = "Sentences.tsv"

# Leer el archivo .tsv con pandas
data = pd.read_csv(file_path, sep='\t')

# Mostrar las primeras filas del archivo
print("Primeras filas del archivo:")
print(data.head())

df = data.drop ('1276', axis = 1)

print("Primeras filas del archivo:")
print(df.head())

df = df.drop ('77', axis = 1)

print("Primeras filas del archivo:")
print(df.head())

df.to_csv('SentencesFinal.csv', index=False)



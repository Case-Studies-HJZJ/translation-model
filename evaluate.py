# Cargar el modelo
import pickle
from model import translate_sentence
""""
model = load_model('nmt_model.h5')

# Cargar los tokenizadores
with open('tokenizer_eng.pkl', 'rb') as f:
    tokenizer_eng = pickle.load(f)
with open('tokenizer_ger.pkl', 'rb') as f:
    tokenizer_ger = pickle.load(f)
"""
test_sentences = [
    "hello",
    "good morning",
    "how are you?"
]


# Traducir y mostrar los resultados
for sentence in test_sentences:
    print(f"Input: {sentence}")
    print(f"Translation: {translate_sentence(sentence)}")
    print()
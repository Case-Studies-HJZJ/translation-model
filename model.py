import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load the dataset from a csv file
data = pd.read_csv("SentencesFinal.csv")

# Separate the columns of sentences
english_sentences = data['English'].tolist()
german_sentences = data['German'].tolist()

# Adding additional tokens for the german sentences
german_sentences = ["<start> " + sentence + " <end>" for sentence in german_sentences]

# Parameters
num_words = 1000
max_sequence_length = 10

# Tokenizer for english
tokenizer_eng = Tokenizer(num_words=num_words, filters='')
tokenizer_eng.fit_on_texts(english_sentences)
input_sequences = tokenizer_eng.texts_to_sequences(english_sentences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

# Tokenizer for german
tokenizer_ger = Tokenizer(num_words=num_words, filters='')
tokenizer_ger.fit_on_texts(german_sentences)
target_sequences = tokenizer_ger.texts_to_sequences(german_sentences)
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

# Getting size of vocabulary
input_vocab_size = len(tokenizer_eng.word_index) + 1
target_vocab_size = len(tokenizer_ger.word_index) + 1

# Model parameters
embedding_dim = 128
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model definition
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Desplazar la secuencia de salida (y_train)
# Shifting the target sequence
target_sequences_input = target_sequences[:, :-1]
target_sequences_output = target_sequences[:, 1:]

# Añadir dimensión extra para el formato (batch_size, sequence_length, 1)
# Extra dimension for the format 
target_sequences_output = np.expand_dims(target_sequences_output, axis=-1)

# Training
epochs = 100
batch_size = 16
print("Forma de input_sequences:", input_sequences.shape)
print("Forma de target_sequences_input:", target_sequences_input.shape)
print("Forma de target_sequences_output:", target_sequences_output.shape)

# Use only the first 10000 samples
input_sequences = input_sequences[:10000]
target_sequences_input = target_sequences_input[:10000]
target_sequences_output = target_sequences_output[:10000]

"""
dataset = tf.data.Dataset.from_tensor_slices((
    (input_sequences, target_sequences_input),
    target_sequences_output
))

dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

history = model.fit(dataset, epochs=epochs, validation_split=0.2)

"""
history = model.fit(
    [input_sequences, target_sequences_input],
    target_sequences_output,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
) 


# Modelo de inferencia para el encoder
# Inference Model for the encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Inference model for the decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Translate function given an input
def translate_sentence(input_text):
    input_seq = tokenizer_eng.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')

    # Obtener el estado del encoder
    states_value = encoder_model.predict(input_seq)

    # Generar secuencia de inicio para el decoder
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_ger.word_index['<start>']

    translated_sentence = []
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Obtener el token con mayor probabilidad
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_ger.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(translated_sentence) > max_sequence_length:
            stop_condition = True
        else:
            translated_sentence.append(sampled_word)

        # Actualizar la secuencia de entrada y los estados
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(translated_sentence)

# Example
print(translate_sentence("guten"))

# Save model
model.save('nmt_model.h5')

# Import tokenizers
import pickle
with open('tokenizer_eng.pkl', 'wb') as f:
    pickle.dump(tokenizer_eng, f)
with open('tokenizer_ger.pkl', 'wb') as f:
    pickle.dump(tokenizer_ger, f)



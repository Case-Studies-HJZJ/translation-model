import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Preprocessing function
def preprocess_text(sentence):
    return ''.join(char for char in sentence if char.isalnum() or char.isspace()).lower()

# Load the dataset from a CSV file
data = pd.read_csv("data/Informal_Sentences.csv")
english_sentences = data['English'].apply(preprocess_text).tolist()
german_sentences = ["<start> " + preprocess_text(sentence) + " <end>" for sentence in data['German'].tolist()]

# Parameters
num_words = 5000  # Increased vocabulary size
max_sequence_length = 20  # Allow longer sequences

# Tokenizer for English
tokenizer_eng = Tokenizer(num_words=num_words, filters='', oov_token='<unk>')
tokenizer_eng.fit_on_texts(english_sentences)
input_sequences = tokenizer_eng.texts_to_sequences(english_sentences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Tokenizer for German
tokenizer_ger = Tokenizer(num_words=num_words, filters='', oov_token='<unk>')
tokenizer_ger.fit_on_texts(german_sentences)
target_sequences = tokenizer_ger.texts_to_sequences(german_sentences)
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Vocabulary sizes
input_vocab_size = len(tokenizer_eng.word_index) + 1
target_vocab_size = len(tokenizer_ger.word_index) + 1

# Model parameters
embedding_dim = 256
latent_dim = 512

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

# Shift the target sequences
target_sequences_input = target_sequences[:, :-1]
target_sequences_output = target_sequences[:, 1:]
target_sequences_output = np.expand_dims(target_sequences_output, axis=-1)

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Training
epochs = 30  # Train for more epochs
batch_size = 64  # Larger batch size for faster training
history = model.fit(
    [input_sequences, target_sequences_input],
    target_sequences_output,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
model.save('nmt_model.keras')

# Save tokenizers using pickle
with open('tokenizer_eng.pkl', 'wb') as f:
    pickle.dump(tokenizer_eng, f)
with open('tokenizer_ger.pkl', 'wb') as f:
    pickle.dump(tokenizer_ger, f)

# Inference Models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_output = decoder_embedding(decoder_inputs)
decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding_output, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Translate function
def translate_sentence(input_text):
    input_seq = tokenizer_eng.texts_to_sequences([preprocess_text(input_text)])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_ger.word_index['<start>']

    translated_sentence = []
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_ger.index_word.get(sampled_token_index, '<unk>')

        if sampled_word == '<end>' or len(translated_sentence) > max_sequence_length:
            stop_condition = True
        else:
            translated_sentence.append(sampled_word)

        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(translated_sentence)

# Example translation
print(translate_sentence("hello, how are you today?"))
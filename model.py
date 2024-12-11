import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Preprocessing function
def preprocess_text(sentence):
    if not isinstance(sentence, str):
        return ""
    return ''.join(char for char in sentence if char.isalnum() or char.isspace()).lower()

# Load the datasets
informal_data = pd.read_csv("data/Informal_Sentences.csv")
formal_data = pd.read_csv("data/Formal_Sentences.csv")

# Prepare sentences for both datasets
informal_english_sentences = informal_data['English'].apply(preprocess_text).tolist()
informal_german_sentences = ["<start> " + preprocess_text(sentence) + " <end>" for sentence in informal_data['German'].tolist()]

formal_english_sentences = formal_data['English'].apply(preprocess_text).tolist()
formal_german_sentences = ["<start> " + preprocess_text(sentence) + " <end>" for sentence in formal_data['German'].tolist()]

# Parameters
num_words = 15000
max_sequence_length = 20
embedding_dim = 256
latent_dim = 512

# Tokenizers and sequences for Informal data
informal_tokenizer_eng = Tokenizer(num_words=num_words, filters='', oov_token='<unk>')
informal_tokenizer_eng.fit_on_texts(informal_english_sentences)
informal_input_sequences = informal_tokenizer_eng.texts_to_sequences(informal_english_sentences)
informal_input_sequences = pad_sequences(informal_input_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

informal_tokenizer_ger = Tokenizer(num_words=num_words, filters='', oov_token='<unk>')
informal_tokenizer_ger.fit_on_texts(informal_german_sentences)
informal_target_sequences = informal_tokenizer_ger.texts_to_sequences(informal_german_sentences)
informal_target_sequences = pad_sequences(informal_target_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Tokenizers and sequences for Formal data
formal_tokenizer_eng = Tokenizer(num_words=num_words, filters='', oov_token='<unk>')
formal_tokenizer_eng.fit_on_texts(formal_english_sentences)
formal_input_sequences = formal_tokenizer_eng.texts_to_sequences(formal_english_sentences)
formal_input_sequences = pad_sequences(formal_input_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

formal_tokenizer_ger = Tokenizer(num_words=num_words, filters='', oov_token='<unk>')
formal_tokenizer_ger.fit_on_texts(formal_german_sentences)
formal_target_sequences = formal_tokenizer_ger.texts_to_sequences(formal_german_sentences)
formal_target_sequences = pad_sequences(formal_target_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Vocabulary sizes
informal_input_vocab_size = min(len(informal_tokenizer_eng.word_index) + 1, num_words)
informal_target_vocab_size = min(len(informal_tokenizer_ger.word_index) + 1, num_words)

formal_input_vocab_size = min(len(formal_tokenizer_eng.word_index) + 1, num_words)
formal_target_vocab_size = min(len(formal_tokenizer_ger.word_index) + 1, num_words)

# Model definition is shared for both datasets
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(informal_input_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_embedding = BatchNormalization()(encoder_embedding)
encoder_lstm = LSTM(latent_dim, return_state=True, dropout=0.3, recurrent_dropout=0.3)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(informal_target_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_embedding = BatchNormalization()(decoder_embedding)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(informal_target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Shift target sequences for training
informal_target_sequences_input = informal_target_sequences[:, :-1]
informal_target_sequences_output = informal_target_sequences[:, 1:]
informal_target_sequences_output = np.expand_dims(informal_target_sequences_output, axis=-1)

formal_target_sequences_input = formal_target_sequences[:, :-1]
formal_target_sequences_output = formal_target_sequences[:, 1:]
formal_target_sequences_output = np.expand_dims(formal_target_sequences_output, axis=-1)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
checkpoint = ModelCheckpoint(
    filepath='nmt_checkpoint_model_{epoch:02d}-{val_loss:.2f}.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Training for Informal dataset
epochs = 30
batch_size = 64
informal_history = model.fit(
    [informal_input_sequences, informal_target_sequences_input],
    informal_target_sequences_output,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Save the final model for Informal data
model.save('nmt_informal_model_final.keras')

# Save training history for Informal data
with open('informal_training_history.pkl', 'wb') as f:
    pickle.dump(informal_history.history, f)

# Training for Formal dataset
formal_history = model.fit(
    [formal_input_sequences, formal_target_sequences_input],
    formal_target_sequences_output,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Save the final model for Formal data
model.save('nmt_formal_model_final.keras')

# Save training history for Formal data
with open('formal_training_history.pkl', 'wb') as f:
    pickle.dump(formal_history.history, f)

# Save tokenizers
with open('informal_tokenizer_eng.pkl', 'wb') as f:
    pickle.dump(informal_tokenizer_eng, f)
with open('informal_tokenizer_ger.pkl', 'wb') as f:
    pickle.dump(informal_tokenizer_ger, f)

with open('formal_tokenizer_eng.pkl', 'wb') as f:
    pickle.dump(formal_tokenizer_eng, f)
with open('formal_tokenizer_ger.pkl', 'wb') as f:
    pickle.dump(formal_tokenizer_ger, f)

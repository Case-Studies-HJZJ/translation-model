import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preprocessing function
def preprocess_text(sentence):
    return ''.join(char for char in sentence if char.isalnum() or char.isspace()).lower()

# Parameters
max_sequence_length = 20
embedding_dim = 256
latent_dim = 512
latest_checkpoint = 'nmt_checkpoint_model.keras'  # Path to your saved checkpoint

# Load the saved tokenizers
with open('tokenizer_eng.pkl', 'rb') as f:
    tokenizer_eng = pickle.load(f)
with open('tokenizer_ger.pkl', 'rb') as f:
    tokenizer_ger = pickle.load(f)

# Vocabulary sizes
input_vocab_size = len(tokenizer_eng.word_index) + 1
target_vocab_size = len(tokenizer_ger.word_index) + 1

# Rebuild the encoder model
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding_layer = Embedding(input_vocab_size, embedding_dim)
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

# Rebuild the decoder model
decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(target_vocab_size, embedding_dim)
decoder_embedding_output = decoder_embedding_layer(decoder_inputs)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_lstm_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding_output, initial_state=decoder_states_inputs
)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_lstm_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + [state_h, state_c]
)

# Load weights from the latest checkpoint
encoder_model.load_weights(latest_checkpoint, by_name=True)
decoder_model.load_weights(latest_checkpoint, by_name=True)

# Translate function
def translate_sentence(input_text):
    # Preprocess input
    input_seq = tokenizer_eng.texts_to_sequences([preprocess_text(input_text)])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')

    # Get the encoder's states
    states_value = encoder_model.predict(input_seq)

    # Initialize the target sequence with the <start> token
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

        # Update the target sequence and states
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(translated_sentence)

# Test the translation
input_sentence = "do you like to go to school?"
print("Input:", input_sentence)
print("Translation:", translate_sentence(input_sentence))

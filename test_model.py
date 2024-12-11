import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load tokenizers and models
with open('informal_tokenizer_eng.pkl', 'rb') as f:
    informal_tokenizer_eng = pickle.load(f)
with open('informal_tokenizer_ger.pkl', 'rb') as f:
    informal_tokenizer_ger = pickle.load(f)

with open('formal_tokenizer_eng.pkl', 'rb') as f:
    formal_tokenizer_eng = pickle.load(f)
with open('formal_tokenizer_ger.pkl', 'rb') as f:
    formal_tokenizer_ger = pickle.load(f)

informal_model = load_model('nmt_informal_model_final.keras')
formal_model = load_model('nmt_formal_model_final.keras')

# Define preprocess_text for safety
def preprocess_text(sentence):
    return ''.join(char for char in sentence if char.isalnum() or char.isspace()).lower()

# Define inference functions
def translate_sentence(input_text, tokenizer_eng, tokenizer_ger, model):
    max_sequence_length = model.input_shape[0][1]  # Assumes input shape is consistent
    
    input_seq = tokenizer_eng.texts_to_sequences([preprocess_text(input_text)])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_ger.word_index['<start>']

    translated_sentence = []
    stop_condition = False

    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_ger.index_word.get(sampled_token_index, '<unk>')

        if sampled_word == '<end>' or len(translated_sentence) >= max_sequence_length:
            stop_condition = True
        else:
            translated_sentence.append(sampled_word)
            target_seq = np.array([[sampled_token_index]])

    return ' '.join(translated_sentence)

# Define BLEU evaluation
def evaluate_translation(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothing_fn = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_fn)

# Example evaluation
if __name__ == "__main__":
    # Informal sentence evaluation
    input_informal = "hello, how are you today?"
    reference_informal = "hallo wie geht es dir heute"
    informal_translation = translate_sentence(input_informal, informal_tokenizer_eng, informal_tokenizer_ger, informal_model)
    informal_bleu = evaluate_translation(reference_informal, informal_translation)

    print(f"Informal Translation: {informal_translation}")
    print(f"Informal BLEU Score: {informal_bleu:.2f}")

    # Formal sentence evaluation
    input_formal = "Good morning, how do you do?"
    reference_formal = "Guten Morgen, wie geht es Ihnen?"
    formal_translation = translate_sentence(input_formal, formal_tokenizer_eng, formal_tokenizer_ger, formal_model)
    formal_bleu = evaluate_translation(reference_formal, formal_translation)

    print(f"Formal Translation: {formal_translation}")
    print(f"Formal BLEU Score: {formal_bleu:.2f}")

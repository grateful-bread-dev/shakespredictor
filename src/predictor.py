from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def generate_text(model, tokenizer, seed_text, max_len, num_words=10):
    """Generate text based on the seed text."""
    # Convert seed text to lowercase and split into words
    seed_text = seed_text.lower()
    seed_words = seed_text.split()
    
    # If seed text is shorter than max_len, pad it
    if len(seed_words) < max_len:
        seed_words = [''] * (max_len - len(seed_words)) + seed_words
    
    # Keep only the last max_len words
    seed_words = seed_words[-max_len:]
    
    for _ in range(num_words):
        # Convert seed words to sequence
        token_list = tokenizer.texts_to_sequences([' '.join(seed_words)])[0]
        
        # Predict next word
        predicted = model.predict(np.array([token_list]), verbose=0)
        predicted_word_idx = predicted.argmax(axis=-1)[0]
        
        # Convert prediction to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_idx:
                output_word = word
                break
        
        # Update seed words
        seed_words = seed_words[1:] + [output_word]
        seed_text += " " + output_word
    
    return seed_text 
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seed_text, max_len, num_words=10):
    """Generate text based on the seed text."""
    for _ in range(num_words):
        # Convert seed text to sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len, padding='post')
        
        # Predict next word
        predicted = model.predict(token_list, verbose=0)
        predicted_word_idx = predicted.argmax(axis=-1)[0]
        
        # Convert prediction to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_idx:
                output_word = word
                break
        
        seed_text += " " + output_word
    
    return seed_text 
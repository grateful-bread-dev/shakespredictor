import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_model(vocab_size, max_len):
    """Create and return the LSTM model."""
    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_len),
        LSTM(64),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, model_path='models/king_lear_model'):
    """Save the trained model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_saved_model(model_path='models/king_lear_model'):
    """Load a saved model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")
    return load_model(model_path) 
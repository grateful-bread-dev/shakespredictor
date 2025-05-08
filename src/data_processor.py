import pdfplumber
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_text_from_pdf(pdf_path):
    """Load and extract text from PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
                
            return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def prepare_sequences(text, sequence_length=10):
    """Prepare input and output sequences from text using sliding window."""
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input")
    
    # Clean and tokenize text
    words = text.lower().split()
    
    if len(words) < sequence_length + 1:
        raise ValueError(f"Text must contain at least {sequence_length + 1} words")
    
    # Create tokenizer and fit on words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([words])
    
    # Convert words to sequences
    sequences = tokenizer.texts_to_sequences([words])[0]
    
    # Create input sequences and labels
    input_sequences = []
    output_sequences = []
    
    for i in range(len(sequences) - sequence_length):
        input_sequences.append(sequences[i:i + sequence_length])
        output_sequences.append(sequences[i + sequence_length])
    
    # Convert to numpy arrays
    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)
    
    return input_sequences, output_sequences, tokenizer, sequence_length 
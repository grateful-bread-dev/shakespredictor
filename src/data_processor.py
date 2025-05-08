import pdfplumber
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def prepare_sequences(text):
    """Prepare input and output sequences from text."""
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input")
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) < 2:
        raise ValueError("Text must contain at least 2 lines for training")
    
    input_lines = lines[:-1]
    output_lines = lines[1:]
    
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input_lines + output_lines)
    
    # Convert to sequences
    input_seq = tokenizer.texts_to_sequences(input_lines)
    output_seq = tokenizer.texts_to_sequences(output_lines)
    
    # Pad sequences
    max_len = max(len(seq) for seq in input_seq + output_seq)
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    output_seq = pad_sequences(output_seq, maxlen=max_len, padding='post')
    
    return input_seq, output_seq, tokenizer, max_len 
import pdfplumber

with pdfplumber.open("king_lear.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

lines = [line.strip() for line in text.split('\n') if line.strip()]
input_lines = lines[:-1]
output_lines = lines[1:]        

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_lines + output_lines)
input_seq = tokenizer.texts_to_sequences(input_lines)
output_seq = tokenizer.texts_to_sequences(output_lines)

max_len = max(len(seq) for seq in input_seq + output_seq)
input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
output_seq = pad_sequences(output_seq, maxlen=max_len, padding='post')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    LSTM(64),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
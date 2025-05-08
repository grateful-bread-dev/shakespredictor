# King Lear Text Predictor

A neural network model that learns from Shakespeare's King Lear to predict the next line of text.

## Project Structure

```
shakespredictor/
├── data/
│   └── king_lear.pdf      # The source text
├── src/
│   ├── data_processor.py  # Handles PDF processing and text preparation
│   ├── model.py          # Defines the neural network architecture
│   ├── predictor.py      # Contains text generation functions
│   └── main.py          # Main script to run the project
└── README.md
```

## Requirements

- Python 3.x
- TensorFlow
- pdfplumber

## Usage

1. Place the King Lear PDF in the `data` directory
2. Run the main script:
   ```bash
   python src/main.py
   ```

The script will:
1. Load and process the text
2. Train the model
3. Generate sample predictions

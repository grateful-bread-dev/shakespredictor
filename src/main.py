import os
import argparse
from data_processor import load_text_from_pdf, prepare_sequences
from model import create_model, save_model, load_saved_model
from predictor import generate_text

def main():
    parser = argparse.ArgumentParser(description='Train or use the King Lear text predictor')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--seed', type=str, default="KING LEAR", help='Seed text for generation')
    parser.add_argument('--words', type=int, default=20, help='Number of words to generate')
    args = parser.parse_args()

    model_path = 'models/king_lear_model'
    
    if args.train or not os.path.exists(model_path):
        try:
            # Load and prepare data
            print("Loading and preparing data...")
            text = load_text_from_pdf("data/king_lear.pdf")
            input_seq, output_seq, tokenizer, max_len = prepare_sequences(text)
            
            # Create and train model
            print("Creating and training model...")
            vocab_size = len(tokenizer.word_index) + 1
            model = create_model(vocab_size, max_len)
            
            history = model.fit(
                input_seq, 
                output_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Save the trained model
            save_model(model, model_path)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return
    else:
        try:
            # Load the saved model
            print("Loading saved model...")
            model = load_saved_model(model_path)
            
            # We need to recreate the tokenizer and get max_len
            # This is a limitation of the current implementation
            text = load_text_from_pdf("data/king_lear.pdf")
            _, _, tokenizer, max_len = prepare_sequences(text)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
    
    # Generate predictions
    print("\nGenerating predictions...")
    print(f"Starting with: {args.seed}")
    try:
        generated = generate_text(model, tokenizer, args.seed, max_len, num_words=args.words)
        print(f"Generated text: {generated}")
    except Exception as e:
        print(f"Error generating text: {str(e)}")

if __name__ == "__main__":
    main() 
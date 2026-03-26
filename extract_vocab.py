"""
Phase 3: Android Vocabulary Extraction
--------------------------------------
Modern Hugging Face pipelines use 'Fast Tokenizers' which hide the vocabulary 
inside a unified JSON file. Android's TFLite libraries require a raw, 
line-by-line 'vocab.txt' file. This script extracts it.
"""

from transformers import DistilBertTokenizer

print("--- Attempting Vocab Extraction ---")

try:
    # We force the use of the "slow" base class to easily access the internal dictionary
    model_path = './saved_model_weights'
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    # Extract the vocabulary and sort by Token ID (0 to 30521)
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    
    # Write the sorted tokens to a plain text file
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for token, _ in sorted_vocab:
            f.write(token + '\n')
            
    print("✅ SUCCESS! 'vocab.txt' is ready for your Android assets folder.")

except Exception as e:
    print(f"❌ Extraction failed: {e}")
    print("Ensure your saved model folder contains the tokenizer config files.")

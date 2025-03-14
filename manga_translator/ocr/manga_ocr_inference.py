import onnxruntime
import numpy as np
from PIL import Image
import os
import re
import jaconv

class MangaOCR:
    def __init__(self, model_path, vocab_path):
        """Initialize the Manga OCR model.
        
        Args:
            model_path (str): Path to the ONNX model file
            vocab_path (str): Path to the vocabulary file
        """
        # Initialize ONNX Runtime session
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=['OpenVINOExecutionProvider']  # Use CPU provider for compatibility
        )
        
        # Load vocabulary
        self.vocab = self._load_vocab(vocab_path)
        
    def _load_vocab(self, vocab_path):
        """Load vocabulary from file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def _preprocess(self, image_path):
        """Preprocess the image for model input.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image tensor
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image = image.convert("L").convert("RGB")  # Convert to grayscale then RGB
        image = image.resize((224, 224), resample=2)  # Use bilinear resampling
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array /= 255.0  # Scale to [0, 1]
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def _generate(self, image_tensor, max_length=300):
        """Generate text tokens from image.
        
        Args:
            image_tensor (numpy.ndarray): Preprocessed image tensor
            max_length (int): Maximum length of generated sequence
            
        Returns:
            list: Generated token IDs
        """
        token_ids = [2]  # Start with token ID 2
        
        for _ in range(max_length):
            # Prepare token ids input
            token_ids_array = np.array([token_ids], dtype=np.int64)
            
            # Run inference
            inputs = {
                'image': image_tensor,
                'token_ids': token_ids_array
            }
            outputs = self.session.run(['logits'], inputs)
            
            # Get logits from the last position
            logits = outputs[0][0, -1]
            
            # Get the token with highest probability
            next_token = int(np.argmax(logits))
            token_ids.append(next_token)
            
            # Stop if end token is generated
            if next_token == 3:
                break
                
        return token_ids
    
    def _decode(self, token_ids):
        """Decode token IDs to text.
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            str: Decoded text
        """
        text = ''
        for token_id in token_ids:
            if token_id < 5:  # Skip special tokens
                continue
            text += self.vocab[token_id]
        return text
    
    def _postprocess(self, text):
        """Postprocess the decoded text.
        
        Args:
            text (str): Decoded text
            
        Returns:
            str: Processed text
        """
        text = "".join(text.split())  # Remove spaces
        text = text.replace("…", "...")  # Replace ellipsis
        text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)  # Normalize dots
        text = jaconv.h2z(text, ascii=True, digit=True)  # Convert to full-width
        return text
    
    def __call__(self, image_path):
        """Run inference on an image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Detected text
        """
        # Preprocess image
        image_tensor = self._preprocess(image_path)
        
        # Generate tokens
        token_ids = self._generate(image_tensor)
        
        # Decode to text and postprocess
        text = self._decode(token_ids)
        text = self._postprocess(text)
        
        return text

def main():
    # Example usage
    model_path = "./models/ocr/quantized_model.onnx"
    vocab_path = "./models/ocr/vocab.txt"  # Make sure to provide the correct path to vocab.txt
    
    # Initialize model
    ocr = MangaOCR(model_path, vocab_path)
    
    # Run inference on a test image
    test_image = "./result/ocrs/37.png"  # Replace with your test image path
    if os.path.exists(test_image):
        result = ocr(test_image)
        print(f"Detected text: {result}")
    else:
        print(f"Test image not found: {test_image}")

if __name__ == "__main__":
    main()

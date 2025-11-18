import open_clip
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional

# Global cache for models and preprocessors to avoid reloading
_MODEL_CACHE = {}

def load_clip_model(
    model_name: str = "ViT-B/32", 
    pretrained: str = "openai", 
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, callable, callable]:
    """
    Load a pre-trained CLIP model, its tokenizer, and preprocessing function.
    Caches the model and preprocessor to avoid reloading.

    Args:
        model_name: Model architecture (e.g., "ViT-B/32", "ViT-L/14", "RN50").
        pretrained: Pre-trained weights to use (e.g., "openai", "laion2b_s34b_b79k").
        device: Device to load the model on ("cuda" or "cpu"). Auto-detects if None.

    Returns:
        model: The loaded CLIP model.
        tokenizer: The corresponding tokenizer for text encoding.
        preprocess: The image preprocessing function.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cache_key = (model_name, pretrained, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    # Tokenizer might not be strictly needed for image-only tasks but good to have
    tokenizer = open_clip.get_tokenizer(model_name) 
    
    _MODEL_CACHE[cache_key] = (model, tokenizer, preprocess)
    return model, tokenizer, preprocess

def preprocess_images_batch(
    images: Union[Image.Image, List[Image.Image]], 
    preprocess_fn: callable,
) -> torch.Tensor:
    """
    Preprocess one or more images for CLIP encoding using a provided preprocessing function.

    Args:
        images: A single PIL image or a list of PIL images.
        preprocess_fn: The preprocessing function obtained from loading the model.

    Returns:
        tensor: Preprocessed image tensor(s) ready for model input.
    """
    if isinstance(images, Image.Image):
        # Unsqueeze to add batch dimension
        return preprocess_fn(images).unsqueeze(0)  
    
    # Handle list of images
    # This stacks them into a single tensor for batch processing
    batch = torch.stack([preprocess_fn(img) for img in images])
    return batch

def extract_image_embeddings_batch(
    model: torch.nn.Module, 
    preprocessed_images_tensor: torch.Tensor, 
    normalize: bool = True, 
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Extract embeddings from a batch of preprocessed images using CLIP model.

    Args:
        model: Loaded CLIP model.
        preprocessed_images_tensor: Tensor of preprocessed images from preprocess_images_batch.
        normalize: Whether to L2-normalize the output embeddings (recommended).
        device: Device to perform inference on. Auto-detects if None.

    Returns:
        embeddings: Torch tensor of image feature embeddings.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
            
    preprocessed_images_tensor = preprocessed_images_tensor.to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_images_tensor)
        
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
    return image_features

def get_image_embeddings(
    images: Union[Image.Image, List[Image.Image]],
    model_name: str = "ViT-B/32",
    pretrained: str = "openai",
    normalize: bool = True,
    device: Optional[str] = None
) -> np.ndarray:
    """
    End-to-end function to load model (or use cached), preprocess images, 
    and extract CLIP embeddings.

    Args:
        images: A single PIL image or a list of PIL images.
        model_name: CLIP model architecture to use.
        pretrained: Pre-trained weights to use.
        normalize: Whether to L2-normalize the embeddings.
        device: Device to perform inference on. Auto-detects if None.

    Returns:
        embeddings: NumPy array of image embeddings.
    """
    if device is None:
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_to_use = device

    # Load model and preprocessing function (uses cache)
    model, _, preprocess_fn = load_clip_model(model_name, pretrained, device_to_use)
    
    # Preprocess images
    preprocessed_tensor = preprocess_images_batch(images, preprocess_fn)
    
    # Extract embeddings
    embeddings_tensor = extract_image_embeddings_batch(model, preprocessed_tensor, normalize, device_to_use)
        
    return embeddings_tensor.cpu().numpy()
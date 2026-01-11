import torch
from sentence_transformers import SentenceTransformer
import numpy
import typing

# Use CUDA if available, otherwise fallback to CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sbert_model = None

def setup(device: str):
    global sbert_model
    # set up model
    sbert_model = SentenceTransformer('all-mpnet-base-v2', device = device)

def generate_embedding(text: typing.Union[str, typing.List[str]]) -> numpy.ndarray:
    global sbert_model
    if sbert_model is None: 
        setup(device)
    return sbert_model.encode(text)
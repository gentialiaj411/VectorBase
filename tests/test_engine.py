import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from minivector.binary_engine import BinaryIndex

def test_binary_quantization():
    # Create random float vectors
    vectors = np.random.randn(100, 384).astype('float32')
    
    # Manually quantize
    bits = (vectors > 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    
    # Check shape (384 bits = 48 bytes)
    assert packed.shape == (100, 48)

def test_index_initialization():
    index = BinaryIndex(dimension=384)
    assert index.dimension == 384
    assert index.binary_vectors is None

def test_search_logic():
    # Mock index
    index = BinaryIndex(dimension=8)
    # 2 vectors: [1,1,1,1,0,0,0,0], [0,0,0,0,1,1,1,1]
    # Packed: [240], [15] (approx)
    
    vec1 = np.array([1,1,1,1,0,0,0,0], dtype='float32')
    vec2 = np.array([0,0,0,0,1,1,1,1], dtype='float32')
    
    index.original_vectors = np.vstack([vec1, vec2])
    index.metadata = [{'id': '1'}, {'id': '2'}]
    
    # Quantize
    bits = (index.original_vectors > 0).astype(np.uint8)
    index.binary_vectors = np.packbits(bits, axis=1)
    
    # Search for vec1
    results = index.search(vec1, k=1)
    assert results[0]['id'] == '1'
    assert results[0]['score'] == 1.0

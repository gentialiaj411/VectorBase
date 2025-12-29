import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from minivector.binary_engine import BinaryIndex
def test_binary_quantization():
    vectors = np.random.randn(100, 384).astype('float32')
    bits = (vectors > 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    assert packed.shape == (100, 48)
def test_index_initialization():
    index = BinaryIndex(dimension=384)
    assert index.dimension == 384
    assert index.binary_vectors is None
def test_search_logic():
    index = BinaryIndex(dimension=8)
    vec1 = np.array([1,1,1,1,0,0,0,0], dtype='float32')
    vec2 = np.array([0,0,0,0,1,1,1,1], dtype='float32')
    index.original_vectors = np.vstack([vec1, vec2])
    index.metadata = [{'id': '1'}, {'id': '2'}]
    bits = (index.original_vectors > 0).astype(np.uint8)
    index.binary_vectors = np.packbits(bits, axis=1)
    results = index.search(vec1, k=1)
    assert results[0]['id'] == '1'
    assert results[0]['score'] == 1.0

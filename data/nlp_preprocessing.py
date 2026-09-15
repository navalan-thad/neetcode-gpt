import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        vocab = set()
        for pos in positive:
            for word in pos.split():
                vocab.add(word) 

        for neg in negative:
            for word in neg.split():
                vocab.add(word) 

        sorted_vocab = sorted(vocab)
        mapping = {word: i+1 for i, word in enumerate(sorted_vocab)}

        tensors = []
        for sentence in positive+negative:
            encoded = [float(mapping[word]) for word in sentence.split()]
            tensors.append(torch.Tensor(encoded))

        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)

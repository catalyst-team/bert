import unittest

import numpy as np
from numpy.testing import assert_array_equal

from bert_ner.dataset import KeyphrasesDataset

texts = [
    'Ryan is great at researching new technologies and companies and providing the right kind of analysis for PM',
]
keyphrases = [
    ['great at researching', 'right kind of analysis'],
]


class KeyphrasesDatasetTests(unittest.TestCase):
    def test_should_have_cls_id_as_first_token_for_input_ids(self):
        dataset = KeyphrasesDataset(texts, keyphrases)
        input_ids, _, _ = dataset[0]
        self.assertEqual(dataset.cls_vid, input_ids[0])

    def test_input_ids_should_be_padded(self):
        dataset = KeyphrasesDataset(texts, keyphrases)
        input_ids, _, _ = dataset[0]
        self.assertEqual(512, input_ids.size(0))

    def test_for_short_sequence_the_attention_mask_should_mark_meaningful_positions(self):
        dataset = KeyphrasesDataset(texts, keyphrases)
        _, _, attention_mask = dataset[0]
        self.assertEqual(512, attention_mask.size(0))
        self.assertEqual(20, attention_mask.sum())
        self.assertEqual(20, attention_mask[:20].sum())

    def test_labels_should_be_assigned_correctly(self):
        dataset = KeyphrasesDataset(texts, keyphrases)
        _, labels, _ = dataset[0]

        expected = np.zeros(20, dtype=int)
        expected[[3, 4, 5]] = 1
        expected[[13, 14, 15, 16]] = 1
        assert_array_equal(expected, labels[:20].numpy())

    def test_labels_do_not_match_should_not_assign(self):
        dataset = KeyphrasesDataset(texts, [['bla']])
        _, labels, _ = dataset[0]

        self.assertEqual(0, labels.sum())

    def test_labels_match_partially_should_not_assign(self):
        dataset = KeyphrasesDataset(texts, [['researching new amazing technologies']])
        _, labels, _ = dataset[0]

        self.assertEqual(0, labels.sum())

    def test_long_sequences_should_get_truncated(self):
        long_texts = [t * 100 for t in texts]
        dataset = KeyphrasesDataset(long_texts, keyphrases)
        input_ids, _, _ = dataset[0]

        self.assertEqual(512, input_ids.size(0))

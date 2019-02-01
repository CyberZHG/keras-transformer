from unittest import TestCase
from keras_transformer.transformer import _get_max_suffix_repeat_times


class TestSuffixRepeat(TestCase):

    def test_abcd(self):
        self.assertEqual(1, _get_max_suffix_repeat_times('abcdabcdabcd', max_len=3))
        self.assertEqual(1, _get_max_suffix_repeat_times('abcdabcdabcd', max_len=6))
        self.assertEqual(2, _get_max_suffix_repeat_times('abcdabcdabcd', max_len=11))
        self.assertEqual(3, _get_max_suffix_repeat_times('abcdabcdabcd', max_len=12))
        self.assertEqual(3, _get_max_suffix_repeat_times('abcdabcdabcd', max_len=16))
        self.assertEqual(2, _get_max_suffix_repeat_times('bcdabcdabcd', max_len=16))

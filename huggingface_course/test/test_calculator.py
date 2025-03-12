from huggingface_course.smolagent_1 import calculate
import unittest


class TestCalculator(unittest.TestCase):
    def test_calculate(self):
        self.assertEqual(calculate("1 + 1*2 + 3"), 6)

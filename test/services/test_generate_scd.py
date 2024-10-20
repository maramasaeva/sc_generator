# test/test_generate_scd.py
import unittest
from generate_scd import generate_supercollider_code

class TestGenerateSCD(unittest.TestCase):
    def test_code_generation(self):
        prompt = "A calm evening by the lake"
        code = generate_supercollider_code(prompt)
        self.assertIn('SynthDef', code)  # Check if code contains expected elements

if __name__ == '__main__':
    unittest.main()
import unittest

import app


class AppTests(unittest.TestCase):
    def test_main_exists(self):
        self.assertTrue(callable(app.main))

    def test_predict_next_word_returns_word_from_vocabulary(self):
        seed = "i like machine learning"
        prediction = app.predict_next_word(seed)
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, app.vocabulary)


if __name__ == "__main__":
    unittest.main()
"""Test if the necessary files exit"""

from train_model import DIRECTORY
from train_model import CATEGORIES
import unittest
import os


class TestFileExists(unittest.TestCase):
    """Test if dataset exists"""

    def test_dataset_exists(self):
        """Test if dataset, subdirectories of dataset and images in them exist"""

        self.assertTrue(os.path.isdir(DIRECTORY), 'There is no directory dataset!')
        self.assertTrue(os.listdir(DIRECTORY) == list(CATEGORIES),
                        'There is no categories!')
        self.assertTrue(len(os.path.join(DIRECTORY, CATEGORIES[0])) != 0,
                        'There is no images in the first directory!')
        self.assertTrue(len(os.path.join(DIRECTORY, CATEGORIES[1])) != 0,
                        'There is no images in the second directory!')


if __name__ == '__main__':
    unittest.main()

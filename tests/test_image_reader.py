"""
tests/test_image_reader.py
──────────────────────────
Real-OCR tests for src/image_reader.py using drawn Instagram-style screenshots
(see tests/screenshot_fixtures.py). Requires Tesseract to be installed.

Run:  python -m unittest tests.test_image_reader -v     (from the project root)
"""
import os, re, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screenshot_fixtures as S
from src.image_reader import read_profile_images, OCR_AVAILABLE

UI_TEXT = re.compile(r"\b(message|following|follow|followed by|champions|stadium)\b", re.I)


@unittest.skipUnless(OCR_AVAILABLE, "Tesseract / pytesseract not installed")
class ImageReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = S.make_all(tempfile.mkdtemp(prefix="sg_shots_"))
        cls.read = {n: read_profile_images([c["path"]])["fields"] for n, c in cls.cases.items()}

    def test_bio_length_matches_truth(self):
        for name, c in self.cases.items():
            t = c["truth"]
            if t["bio_len"] is None:
                continue
            with self.subTest(name):
                self.assertEqual(self.read[name]["bio_len"]["value"], t["bio_len"])

    def test_bio_never_contains_ui_or_caption_text(self):
        """The original bug: buttons / 'Followed by' / post captions leaking in as the 'bio'."""
        for name, f in self.read.items():
            with self.subTest(name):
                self.assertIsNone(UI_TEXT.search(f["bio_text"]["value"] or ""), f["bio_text"]["value"])

    def test_empty_bio_is_zero_not_invented(self):
        f = self.read["mobile_empty_bio"]
        self.assertEqual(f["bio_len"]["value"], 0)
        self.assertEqual(f["bio_text"]["value"], "")

    def test_truncated_bio_is_flagged_low(self):
        self.assertEqual(self.read["mobile_truncated"]["bio_len"]["confidence"], "low")

    def test_screenshot_without_profile_header_guesses_nothing(self):
        f = self.read["grid_only"]
        self.assertIsNone(f["bio_len"]["value"])
        self.assertEqual(f["bio_len"]["confidence"], "none")

    def test_stats_read_correctly_in_every_layout(self):
        for name, c in self.cases.items():
            if name == "grid_only":
                continue
            t = c["truth"]
            with self.subTest(name):
                for k in ("posts", "followers", "following"):
                    self.assertEqual(self.read[name][k]["display"].replace(" ", ""), t[k])

    def test_dark_mode_is_readable(self):
        self.assertEqual(self.read["mobile_dark"]["followers"]["display"], "48.2K")


if __name__ == "__main__":
    unittest.main(verbosity=2)
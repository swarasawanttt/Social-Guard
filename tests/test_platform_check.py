import unittest
from src.platform_check import check_platform_matches, detect_platform_from_link

class PlatformCheck(unittest.TestCase):
    def bad(self, link, sel, found):
        ok, msg = check_platform_matches(link, sel)
        self.assertFalse(ok, (link, sel)); self.assertIn(found, msg); self.assertIn("selected", msg)
    def good(self, link, sel):
        self.assertEqual(check_platform_matches(link, sel), (True, ""), (link, sel))

    def test_twitter_link_with_instagram_selected(self): self.bad("https://twitter.com/elonmusk", "Instagram", "Twitter/X")
    def test_x_link_with_instagram_selected(self):       self.bad("https://x.com/elonmusk", "Instagram", "Twitter/X")
    def test_instagram_link_with_twitter_selected(self): self.bad("https://www.instagram.com/cristiano/", "Twitter/X", "Instagram")
    def test_facebook_link_with_linkedin_selected(self): self.bad("facebook.com/zuck", "LinkedIn (Academic)", "Facebook")
    def test_linkedin_link_with_facebook_selected(self): self.bad("https://www.linkedin.com/in/someone", "Facebook", "LinkedIn")
    def test_matching_links_ok(self):
        for l, s in [("https://instagram.com/cristiano", "Instagram"), ("https://x.com/a", "Twitter/X"),
                     ("https://twitter.com/a", "Twitter/X"), ("https://www.facebook.com/zuck", "Facebook"),
                     ("https://www.linkedin.com/in/x", "LinkedIn (Academic)"), ("m.facebook.com/zuck", "Facebook"),
                     ("HTTPS://WWW.INSTAGRAM.COM/Cristiano", "Instagram")]:
            self.good(l, s)
    def test_bare_usernames_ok(self):
        for l in ("cristiano", "@cristiano", "john.smith", "x.y_z"): self.good(l, "Instagram")
    def test_unknown_host_left_to_other_validators(self): self.good("https://instagarm.com/cristiano", "Instagram")
    def test_lookalike_host_not_treated_as_real(self):
        self.assertIsNone(detect_platform_from_link("https://notinstagram.com/a"))
        self.assertIsNone(detect_platform_from_link("https://instagram.com.evil.io/a"))
    def test_userinfo_trick(self): self.bad("https://instagram.com@twitter.com/a", "Instagram", "Twitter/X")
    def test_unrecognised_dropdown_value_ok(self): self.good("https://twitter.com/a", "Other")

if __name__ == "__main__":
    unittest.main()
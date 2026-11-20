"""
tests/test_scraper.py
─────────────────────
Offline tests for src/url_resolver.py and src/scraper.py.
No internet needed: pages are realistic fixtures, plus one local HTTP server
to exercise the real gzip/deflate/404-body handling in _http_get().

Run:  python -m unittest tests.test_scraper -v      (from the project root)
"""
import gzip, json, os, sys, tempfile, threading, unittest, zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import scraper
from src.scraper import fetch_profile_data, parse_count
from src.url_resolver import resolve_target

scraper.DEBUG_DIR = tempfile.mkdtemp(prefix="sg_debug_")


# ── helpers ──────────────────────────────────────────────────────────────────
def R(body="", status=200, url="https://www.instagram.com/x/"):
    return {"status": status, "headers": {}, "body": body, "final_url": url, "error": None}


NET_ERR = {"status": 0, "headers": {}, "body": "", "final_url": "", "error": "URLError: down"}


def fake(responses, default=None):
    """responses: {substring-of-url: response}. First match wins."""
    default = default or R("<html><title>Instagram</title></html>")

    def _get(url, headers=None):
        for needle, resp in responses.items():
            if needle in url:
                return resp
        return default
    return _get


def page(head="", body="", scripts=""):
    return f"<!doctype html><html><head><title>t</title>{head}</head><body>{body}{scripts}</body></html>"


def og(desc, title="Cristiano Ronaldo (@cristiano) • Instagram photos and videos"):
    return (f'<meta property="og:title" content="{title}"/>'
            f'<meta property="og:description" content="{desc}"/>')


OG_DESC = ("650M Followers, 560 Following, 3,800 Posts - See Instagram photos and "
           "videos from Cristiano Ronaldo (@cristiano)")

# Legacy shape
LEGACY = {"entry_data": {"ProfilePage": [{"graphql": {"user": {
    "username": "cristiano", "biography": "SIUUUUbscribe to my channel", "full_name": "Cristiano Ronaldo",
    "is_private": False, "is_verified": True,
    "edge_followed_by": {"count": 650123456}, "edge_follow": {"count": 560},
    "edge_owner_to_timeline_media": {"count": 3812}}}}]}}

# Shape the scraper has NEVER seen: different wrapper names, different key
# names, numbers as strings, profile buried deep with other users beside it.
RENAMED = {"props": {"pageProps": {"layout": {"slots": [
    {"suggested": [{"handle": "someoneelse", "followersCount": 999999999,
                    "friendsCount": 1, "mediaCount": 5}]},
    {"hero": {"account": {"handle": "cristiano", "about": "New layout bio here",
                          "followersCount": "650,123,456", "friendsCount": 561,
                          "mediaCount": "3,813"}}}]}}}}


class ResolverTests(unittest.TestCase):
    def test_valid_forms(self):
        for raw, hint in [("https://www.instagram.com/cristiano/", ""), ("instagram.com/cristiano?igsh=1", ""),
                          ("@cristiano", "Instagram"), ("cristiano", "Instagram")]:
            r = resolve_target(raw, hint)
            self.assertTrue(r["ok"], raw)
            self.assertEqual(r["username"], "cristiano")

    def test_typo_domain_gets_suggestion(self):
        r = resolve_target("https://www.instagarm.com/cristiano")
        self.assertFalse(r["ok"]); self.assertEqual(r["error_code"], "wrong_domain")
        self.assertIn("instagram.com", r["suggestion"])

    def test_post_link_is_not_profile(self):
        r = resolve_target("https://www.instagram.com/p/CxYz123/")
        self.assertEqual(r["error_code"], "not_a_profile")

    def test_illegal_usernames(self):
        for raw in ("bad..name", "has space", "trailing.", "x" * 31):
            self.assertEqual(resolve_target(raw, "Instagram")["error_code"], "bad_username", raw)

    def test_short_facebook_vanity_allowed(self):
        self.assertTrue(resolve_target("https://m.facebook.com/zuck")["ok"])

    def test_unsupported_site_and_empty(self):
        self.assertEqual(resolve_target("https://www.tiktok.com/@a")["error_code"], "unsupported_site")
        self.assertEqual(resolve_target("")["error_code"], "empty")


class CountParsing(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(parse_count("1.2M"), (1_200_000, True))
        self.assertEqual(parse_count("3,800"), (3800, False))
        self.assertEqual(parse_count("1,2K"), (1200, True))
        self.assertEqual(parse_count("239.8M"), (239_800_000, True))
        self.assertEqual(parse_count("abc"), (None, False))
        self.assertEqual(scraper.parse_manual_value("500+"), 500)


class ScraperExtraction(unittest.TestCase):
    def go(self, responses, user="cristiano", default=None, platform="Instagram"):
        return fetch_profile_data(user, platform, _http=fake(responses, default))

    # --- success paths ---
    def test_legacy_json_blob(self):
        html = page(scripts=f"<script>window._sharedData = {json.dumps(LEGACY)};</script>")
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertTrue(r["success"], r["error"])
        d = r["data"]
        self.assertEqual((d["followers"], d["following"], d["posts"]), (650123456, 560, 3812))
        self.assertEqual(d["bio_len"], len(LEGACY["entry_data"]["ProfilePage"][0]["graphql"]["user"]["biography"]))
        self.assertTrue(d["is_verified"])

    def test_adaptive_renamed_keys_and_unknown_structure(self):
        html = page(scripts=f'<script type="application/json">{json.dumps(RENAMED)}</script>')
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertTrue(r["success"], r)
        d = r["data"]
        self.assertEqual((d["followers"], d["following"], d["posts"]), (650123456, 561, 3813))
        self.assertEqual(d["bio_len"], len("New layout bio here"))     # NOT the other user's data

    def test_json_serialised_inside_a_string(self):
        inner = json.dumps({"user": {"username": "cristiano", "biography": "hi there",
                                     "follower_count": 100, "following_count": 5, "media_count": 7}})
        html = page(scripts=f'<script type="application/json">{json.dumps({"blob": inner})}</script>')
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertTrue(r["success"], r)
        self.assertEqual(r["data"]["followers"], 100)

    def test_escaped_json_inside_javascript(self):
        inner = json.dumps({"username": "cristiano", "biography": "esc bio", "follower_count": 4242,
                            "following_count": 10, "media_count": 99})
        js = 'requireLazy(["x"],function(){ init("' + inner.replace('"', '\\"') + '"); });'
        r = self.go({"instagram.com/cristiano": R(page(scripts=f"<script>{js}</script>"))})
        self.assertTrue(r["success"], r)
        self.assertEqual((r["data"]["followers"], r["data"]["posts"]), (4242, 99))

    def test_related_profiles_never_contaminate(self):
        blob = {"related": [{"username": "other1", "follower_count": 111, "following_count": 2, "media_count": 3,
                             "biography": "WRONG"}],
                "me": {"username": "cristiano", "follower_count": 5000, "following_count": 50,
                       "media_count": 500, "biography": "mine"}}
        html = page(scripts=f'<script type="application/json">{json.dumps(blob)}</script>')
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertTrue(r["success"], r)
        self.assertEqual((r["data"]["followers"], r["data"]["bio_len"]), (5000, 4))

    def test_json_ld_interaction_statistics(self):
        ld = {"@type": "ProfilePage", "mainEntity": {"@type": "Person", "alternateName": "@cristiano",
              "description": "ld bio!", "interactionStatistic": [
                  {"@type": "InteractionCounter", "interactionType": "https://schema.org/FollowAction",
                   "userInteractionCount": 777},
                  {"@type": "InteractionCounter", "interactionType": "https://schema.org/WriteAction",
                   "userInteractionCount": 12}]}}
        html = page(og("777 Followers, 33 Following, 12 Posts"),
                    scripts=f'<script type="application/ld+json">{json.dumps(ld)}</script>')
        r = self.go({"instagram.com/cristiano": R(html)})
        d = r["data"]
        self.assertTrue(r["success"], r)
        self.assertEqual((d["followers"], d["following"], d["posts"], d["bio_len"]), (777, 33, 12, 7))

    def test_exact_json_beats_rounded_meta_and_marks_high_confidence(self):
        html = page(og(OG_DESC), scripts=f"<script>window._sharedData = {json.dumps(LEGACY)};</script>")
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertTrue(r["success"])
        self.assertEqual(r["data"]["followers"], 650123456)
        self.assertEqual(r["data"]["approximate_fields"], [])
        self.assertEqual(r["data"]["data_confidence"], "high")

    def test_twitter_syndication_next_data(self):
        nd = {"props": {"pageProps": {"timeline": {"entries": [{"content": {"tweet": {"user": {
            "screen_name": "jack", "followers_count": 6000000, "friends_count": 4,
            "statuses_count": 30000, "description": "#bitcoin"}}}}]}}}}
        html = page(scripts=f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(nd)}</script>')
        r = self.go({"syndication.twitter.com": R(html, url="https://syndication.twitter.com/x")},
                    user="jack", platform="Twitter/X")
        self.assertTrue(r["success"], r)
        self.assertEqual((r["data"]["followers"], r["data"]["following"], r["data"]["posts"],
                          r["data"]["bio_len"]), (6000000, 4, 30000, 8))

    # --- partial / low-trust => prefilled manual form, never silent trust ---
    def test_meta_only_gives_prefilled_form_missing_only_bio(self):
        r = self.go({"instagram.com/cristiano": R(page(og(OG_DESC)))})
        self.assertFalse(r["success"]); self.assertTrue(r["manual_required"])
        self.assertEqual(r["error_code"], "incomplete")
        vals = {f["name"]: f["value"] for f in r["manual_fields"]}
        self.assertEqual((vals["followers"], vals["following"], vals["posts"]), (650_000_000, 560, 3800))
        self.assertEqual(vals["bio_len"], ""); self.assertEqual(vals["username_len"], 9)
        self.assertTrue(all(f["prefilled"] for f in r["manual_fields"] if f["name"] != "bio_len"))

    def test_conflicting_sources_are_not_trusted(self):
        blob = {"username": "cristiano", "follower_count": 99_000_000, "following_count": 560,
                "media_count": 3800, "biography": "b"}
        html = page(og(OG_DESC), scripts=f'<script type="application/json">{json.dumps(blob)}</script>')
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertFalse(r["success"])
        self.assertIn("followers", r["report"]["conflicts"])

    def test_unowned_page_text_is_low_trust(self):
        r = self.go({"instagram.com/cristiano": R(page(body="<p>5 followers 3 following 9 posts</p>"))})
        self.assertFalse(r["success"])   # text-only evidence must be confirmed by the user

    # --- why-it-failed codes ---
    def test_http_404_is_not_found_and_not_manual(self):
        r = self.go({"instagram.com/cristiano": R("<h2>Sorry, this page isn't available.</h2>", status=404)})
        self.assertEqual(r["error_code"], "not_found")
        self.assertTrue(r["invalid"]); self.assertFalse(r["manual_required"])
        self.assertIn("doesn't appear to exist", r["error"])

    def test_200_with_page_isnt_available_text_is_not_found(self):
        body = page(body="<h2>Sorry, this page isn't available.</h2><p>The link you followed may be broken.</p>")
        r = self.go({"instagram.com": R(body)})
        self.assertEqual(r["error_code"], "not_found")

    def test_login_wall_is_blocked_not_not_found(self):
        wall = R(page(og("Create an account or log in to Instagram - Share what you're into.", "Instagram"),
                      body="<p>Log in to see photos and videos from your friends</p>"))
        r = self.go({"instagram.com": wall})
        self.assertEqual(r["error_code"], "blocked")
        self.assertTrue(r["manual_required"]); self.assertFalse(r["invalid"])

    def test_redirect_to_login_is_blocked(self):
        r = self.go({"instagram.com": R("<html></html>", url="https://www.instagram.com/accounts/login/?next=/x/")})
        self.assertEqual(r["error_code"], "blocked")

    def test_private_account_without_numbers(self):
        r = self.go({"instagram.com": R(page(body="<p>This account is private</p>"))})
        self.assertEqual(r["error_code"], "private")

    def test_private_account_with_public_counts_still_usable(self):
        blob = {"username": "cristiano", "is_private": True, "biography": "private bio",
                "edge_followed_by": {"count": 300}, "edge_follow": {"count": 200},
                "edge_owner_to_timeline_media": {"count": 12}}
        r = self.go({"instagram.com/cristiano": R(page(scripts=f'<script type="application/json">{json.dumps(blob)}</script>'))})
        self.assertTrue(r["success"], r)
        self.assertTrue(r["data"]["is_private"]); self.assertEqual(r["data"]["followers"], 300)

    def test_layout_changed_saves_debug_html(self):
        weird = page(body="<div id='app'>Loading…</div>")
        r = self.go({"instagram.com": R(weird)})
        self.assertEqual(r["error_code"], "layout_changed")
        self.assertTrue(r["report"]["debug_html"])
        self.assertTrue(os.path.exists(os.path.join(scraper.DEBUG_DIR, r["report"]["debug_html"])))

    def test_network_down(self):
        r = self.go({}, default=NET_ERR)
        self.assertEqual(r["error_code"], "network_error"); self.assertTrue(r["manual_required"])

    def test_invalid_input_never_touches_network(self):
        def boom(*a, **k): raise AssertionError("network used")
        r = fetch_profile_data("https://www.instagarm.com/cristiano", "Instagram", _http=boom)
        self.assertTrue(r["invalid"]); self.assertIn("instagram.com", r["suggestion"])


    # --- regressions from the first live Instagram run ---
    def test_stale_preview_meta_is_outvoted_by_two_live_layers(self):
        """Live run: meta said 649 following, JSON + visible text said 637."""
        blob = {"username": "cristiano", "follower_count": 679908339, "following_count": 637,
                "media_count": 4146, "biography": "Real bio text"}
        html = page(og("680M Followers, 649 Following, 4,146 Posts - See Instagram photos"),
                    body="<p>637 following</p>",
                    scripts=f'<script type="application/json">{json.dumps(blob)}</script>')
        r = self.go({"instagram.com/cristiano": R(html)})
        self.assertTrue(r["success"], r)
        self.assertEqual(r["data"]["following"], 637)
        self.assertEqual(r["data"]["followers"], 679908339)
        self.assertEqual(r["data"]["data_confidence"], "medium")      # dispute caps confidence
        self.assertIn("following", r["report"]["disputes"])

    def test_empty_biography_is_not_trusted_as_zero_length(self):
        """Live run: an empty field produced bio_chars=0 for an account that has a bio."""
        blob = {"username": "cristiano", "follower_count": 100, "following_count": 5,
                "media_count": 9, "biography": ""}
        r = self.go({"instagram.com/cristiano": R(page(scripts=f'<script type="application/json">{json.dumps(blob)}</script>'))})
        self.assertFalse(r["success"])
        vals = {f["name"]: f["value"] for f in r["manual_fields"]}
        self.assertEqual(vals["bio_len"], "")                          # left blank, NOT 0
        self.assertTrue(r["report"]["bio_empty_unverified"])
        self.assertIn("EMPTY bio field", r["error"])

    def test_bio_key_priority_ignores_empty_vague_keys(self):
        blob = {"username": "cristiano", "follower_count": 100, "following_count": 5, "media_count": 9,
                "description": "", "summary": "", "biography": "the real one"}
        r = self.go({"instagram.com/cristiano": R(page(scripts=f'<script type="application/json">{json.dumps(blob)}</script>'))})
        self.assertTrue(r["success"], r)
        self.assertEqual(r["data"]["bio_len"], len("the real one"))

    def test_record_preview_is_exposed_for_diagnosis(self):
        blob = {"username": "cristiano", "follower_count": 100, "following_count": 5, "media_count": 9,
                "biography": "x"}
        r = self.go({"instagram.com/cristiano": R(page(scripts=f'<script type="application/json">{json.dumps(blob)}</script>'))})
        previews = [l["record_preview"] for l in r["report"]["layer_values"] if l["record_preview"]]
        self.assertTrue(previews and previews[0]["follower_count"] == 100)

    def test_health_stats_recorded(self):
        self.assertTrue(scraper.scraper_health())


# ── real HTTP layer: gzip, deflate, 404 body, no brotli advertised ──────────
class _H(BaseHTTPRequestHandler):
    seen_accept_encoding = None
    def log_message(self, *a): pass
    def do_GET(self):
        _H.seen_accept_encoding = self.headers.get("Accept-Encoding")
        body, code, enc = b"hello world", 200, None
        if self.path == "/gz":    body, enc = gzip.compress(b"<p>gzip ok</p>"), "gzip"
        if self.path == "/df":    body, enc = zlib.compress(b"<p>deflate ok</p>"), "deflate"
        if self.path == "/404":   body, code = b"<h2>Sorry, this page isn't available.</h2>", 404
        self.send_response(code)
        if enc: self.send_header("Content-Encoding", enc)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers(); self.wfile.write(body)


class HttpLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _H)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        scraper.MIN_HOST_GAP = 0
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def test_gzip(self):     self.assertIn("gzip ok", scraper._http_get(self.base + "/gz")["body"])
    def test_deflate(self):  self.assertIn("deflate ok", scraper._http_get(self.base + "/df")["body"])
    def test_404_keeps_body(self):
        r = scraper._http_get(self.base + "/404", retries=0)
        self.assertEqual(r["status"], 404); self.assertIn("isn't available", r["body"])
    def test_no_brotli_advertised(self):
        scraper._http_get(self.base + "/gz"); self.assertNotIn("br", _H.seen_accept_encoding)
    def test_connection_refused_is_status_0(self):
        r = scraper._http_get("http://127.0.0.1:1/", timeout=2, retries=0)
        self.assertEqual(r["status"], 0); self.assertTrue(r["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
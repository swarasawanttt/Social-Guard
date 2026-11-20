"""
tests/screenshot_fixtures.py
────────────────────────────
Draws realistic Instagram-style profile screenshots (with KNOWN ground truth)
so image_reader.py can be tested with real Tesseract OCR instead of guesses.

Layouts: mobile (light), mobile (dark mode), desktop web, plus the awkward
cases that broke the old reader: empty bio, multi-line bio, bio collapsed
behind "… more", and a posts-grid screenshot full of caption noise.

make_all(outdir) -> {name: {"path": str, "truth": {...}}}
"""
import os
from PIL import Image, ImageDraw, ImageFont

_DJ = "/usr/share/fonts/truetype/dejavu/"
def F(size, bold=False):
    return ImageFont.truetype(_DJ + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def _mobile(path, *, username, name, category, bio_lines, link, followed_by, posts, followers,
            following, dark=False, more=False, button="Following"):
    W, H = 1080, 2340
    bg, fg, grey, blue = ((0, 0, 0), (245, 245, 245), (150, 150, 150), (90, 160, 255)) if dark else \
                         ((255, 255, 255), (20, 20, 20), (120, 120, 120), (0, 90, 200))
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    d.text((70, 70), username, font=F(52, True), fill=fg)
    d.text((70 + 40 + d.textlength(username, font=F(52, True)), 78), "v", font=F(34), fill=fg)
    d.ellipse((60, 190, 300, 430), fill=(130, 130, 130))
    for cx, num, label in ((470, posts, "posts"), (700, followers, "followers"), (930, following, "following")):
        d.text((cx, 255), num, font=F(50, True), fill=fg, anchor="mm")
        d.text((cx, 322), label, font=F(34), fill=fg, anchor="mm")
    y = 470
    d.text((60, y), name, font=F(40, True), fill=fg); y += 62
    if category:
        d.text((60, y), category, font=F(32), fill=grey); y += 56
    for ln in bio_lines:
        d.text((60, y), ln, font=F(38), fill=fg); y += 58
    if more:
        d.text((60, y), "... more", font=F(38), fill=grey); y += 58
    if link:
        d.text((60, y), link, font=F(38, True), fill=blue); y += 66
    if followed_by:
        d.text((60, y), followed_by, font=F(32), fill=grey); y += 90
    btn = (50, 50, 50) if dark else (230, 230, 230)
    d.rounded_rectangle((60, y, 500, y + 90), 18, fill=btn)
    d.text((280, y + 45), button + " v", font=F(38, True), fill=fg, anchor="mm")
    d.rounded_rectangle((530, y, 960, y + 90), 18, fill=btn)
    d.text((745, y + 45), "Message", font=F(38, True), fill=fg, anchor="mm")
    y += 180
    for i, lab in enumerate(("Champions", "Family", "Travel", "Training")):
        cx = 150 + i * 255
        d.ellipse((cx - 70, y, cx + 70, y + 140), outline=grey, width=4)
        d.text((cx, y + 175), lab, font=F(28), fill=fg, anchor="mm")
    y += 300
    for r in range(2):                                         # posts grid with noise text
        for c in range(3):
            d.rectangle((c * 360 + 2, y + r * 360, c * 360 + 358, y + r * 360 + 358), fill=(90 + 20 * c, 100, 110 + 30 * r))
    d.text((20, y + 20), "Training day with the team at the stadium, what a night it was", font=F(30), fill=(255, 255, 255))
    d.text((20, y + 390), "Thank you all for the amazing support this season guys", font=F(30), fill=(255, 255, 255))
    im.save(path)


def _desktop(path, *, username, name, bio_lines, link, posts, followers, following):
    W, H = 1500, 1300
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((220, 90, 450, 320), fill=(130, 130, 130))
    d.text((560, 90), username, font=F(34), fill=(20, 20, 20))
    d.rounded_rectangle((900, 80, 1040, 135), 12, fill=(230, 230, 230)); d.text((970, 108), "Following", font=F(26, True), fill=(20, 20, 20), anchor="mm")
    d.rounded_rectangle((1060, 80, 1190, 135), 12, fill=(230, 230, 230)); d.text((1125, 108), "Message", font=F(26, True), fill=(20, 20, 20), anchor="mm")
    x = 560
    for num, lab in ((posts, "posts"), (followers, "followers"), (following, "following")):
        s = f"{num} {lab}"
        d.text((x, 175), s, font=F(30), fill=(20, 20, 20)); x += int(d.textlength(s, font=F(30))) + 60
    y = 250
    d.text((560, y), name, font=F(30, True), fill=(20, 20, 20)); y += 52
    for ln in bio_lines:
        d.text((560, y), ln, font=F(30), fill=(20, 20, 20)); y += 46
    if link:
        d.text((560, y), link, font=F(30, True), fill=(0, 55, 107)); y += 70
    y = max(y + 60, 520)
    d.line((220, y, 1280, y), fill=(210, 210, 210), width=2)
    for i, t in enumerate(("POSTS", "REELS", "TAGGED")):
        d.text((470 + i * 250, y + 30), t, font=F(24, True), fill=(20, 20, 20))
    for c in range(3):
        d.rectangle((220 + c * 360, y + 100, 220 + c * 360 + 340, y + 440), fill=(100 + 30 * c, 110, 120))
    im.save(path)


def _grid_only(path):
    """A screenshot of just the posts grid: lots of text, NO profile header."""
    im = Image.new("RGB", (1080, 1600), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, t in enumerate(("What a fantastic evening with all my friends at the beach", "New collection dropping soon, stay tuned everyone!!",
                           "Followed by thousands of fans around the world", "Link in bio for the full workout plan and diet tips")):
        d.text((40, 100 + i * 300), t, font=F(36), fill=(20, 20, 20))
    im.save(path)


CASES = {
    "mobile_multiline": dict(kind="m", username="cristiano", name="Cristiano Ronaldo", category="Athlete",
        bio_lines=["Football player | Father of five", "Founder of CR7 Selection and Crunch Fitness", "Join the journey and never give up"],
        link="hrbl.me/CR7Pro2col", followed_by="Followed by lionelmessi, neymarjr and 12 others",
        posts="3,812", followers="650M", following="560"),
    "mobile_dark": dict(kind="m", username="travel_with_anna", name="Anna Weber", category="Digital creator",
        bio_lines=["Travel photographer based in Berlin", "Booking: hello@annaweber.com"], link="annaweber.com",
        followed_by="Followed by mara_k and 3 others", dark=True, posts="842", followers="48.2K", following="615"),
    "mobile_empty_bio": dict(kind="m", username="user8837261", name="user8837261", category=None, bio_lines=[],
        link=None, followed_by=None, posts="0", followers="3", following="7455", button="Follow"),
    "mobile_one_line": dict(kind="m", username="fit_with_raj", name="Raj Patel", category=None,
        bio_lines=["Certified fitness coach. DM for plans!"], link=None, followed_by=None,
        posts="213", followers="9,481", following="402"),
    "mobile_truncated": dict(kind="m", username="the_bookish_one", name="Priya", category="Writer",
        bio_lines=["Reader. Writer. Coffee addict. Reviewing every", "book I finish and some I abandon halfway through,", "because life is too short for bad novels"],
        link=None, followed_by=None, more=True, posts="377", followers="12.7K", following="889"),
    "desktop": dict(kind="d", username="cristiano", name="Cristiano Ronaldo",
        bio_lines=["Football player | Father of five", "Founder of CR7 Selection and Crunch Fitness"],
        link="hrbl.me/CR7Pro2col", posts="3,812", followers="650M", following="560"),
}


def make_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for name, c in CASES.items():
        c = dict(c); kind = c.pop("kind"); path = os.path.join(outdir, name + ".png")
        (_mobile if kind == "m" else _desktop)(path, **c)
        out[name] = {"path": path, "truth": {
            "bio": " ".join(c["bio_lines"]), "bio_len": len(" ".join(c["bio_lines"])),
            "followers": c["followers"], "following": c["following"], "posts": c["posts"],
            "truncated": c.get("more", False), "empty": not c["bio_lines"]}}
    p = os.path.join(outdir, "grid_only.png"); _grid_only(p)
    out["grid_only"] = {"path": p, "truth": {"bio": None, "bio_len": None}}
    return out


if __name__ == "__main__":
    import sys
    print(make_all(sys.argv[1] if len(sys.argv) > 1 else "fixtures_out"))
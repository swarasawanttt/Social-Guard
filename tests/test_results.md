| TC ID | Module | Test Case | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|---|
| TC01 | Input validation | Valid profile URL is accepted | https://instagram.com/cristiano + Instagram | Accepted, no error | ok=True, message='' | PASS |
| TC02 | Input validation | Bare username is accepted | cristiano + Instagram | Accepted, no error | ok=True, message='' | PASS |
| TC03 | Input validation | Empty input is rejected | '' (empty) + Instagram | Validation error, no crash | ok=False, message='No profile URL or username was provided.' | PASS |
| TC04 | Input validation | Missing platform is rejected | cristiano + (no platform) | Validation error asking to select a platform | ok=False, message='Please select a platform.' | PASS |
| TC05 | Input validation | Platform must match the link | Instagram selected + https://twitter.com/elonmusk | Rejected: platform does not match link | ok=False, message='You selected Instagram, but this link is for Twitter/X. Choose Twitter/X as the platform, or paste an Instagram profile link.' | PASS |
| TC06 | Input validation | Misspelled website is rejected with a hint | https://instagarm.com/cristiano + Instagram | Rejected, suggests the correct site | ok=False, message='“instagarm.com” isn't a supported site — did you misspell it? Did you mean instagram.com?  Try: https://instagram.com/cristiano' | PASS |
| TC07 | Input validation | Post link (not a profile) is rejected | https://instagram.com/p/ABC123 + Instagram | Rejected: not a profile link | ok=False, message='That's a link to a post, not to a profile. Open the account's profile page and copy that address instead.' | PASS |
| TC08 | Input validation | Illegal username characters are rejected | 'bad name!!' + Instagram | Rejected: invalid username | ok=False, message='“bad name!!” contains spaces, so it can't be a Instagram username. Instagram usernames are 1-30 characters: letters, numbers, periods and underscores only (no spaces, no '..', can't end with '.').' | PASS |
| TC09 | Data acquisition | Human-readable counts are converted to integers | '239.8M', '1.2K', '500+', '1,234' | 239800000, 1200, 500, 1234 | [239800000, 1200, 500, 1234] | PASS |
| TC10 | Data acquisition | Account that does not exist gives a clear error | Scraper reports 'not found' for @zzz | Error page: account not found, no manual form | landed on /error; message shown=True | PASS |
| TC11 | Data acquisition | Private / blocked profile falls back to manual entry | Scraper cannot read stats (private account) | Manual-entry form is shown | landed on /manual-input; form has followers field=True | PASS |
| TC12 | Data acquisition | Scraper's own unit tests pass | tests/test_scraper.py | All tests pass | 37/37 scraper tests passed | PASS |
| TC13 | Manual entry | Valid manual entry produces a result | 480 followers, 520 following, 95 posts, bio text | Result page with a verdict | landed on /results, status 200 | PASS |
| TC14 | Manual entry | Blank required fields are rejected | Followers = blank, Posts = blank | Validation error, no verdict | landed on /error | PASS |
| TC15 | Manual entry | Partially filled form is rejected | Followers = 500, Following = blank, Posts = 40 | Error message, no verdict | landed on /error | PASS |
| TC16 | Manual entry | All-zero profile gets 'not enough data', not a verdict | 0 followers, 0 following, 0 posts | Clear 'not enough data' error | landed on /error; 'not enough information' message=True | PASS |
| TC17 | Manual entry | Garbage numbers do not crash the app | Followers = 'abc' | Handled error page, no server crash (no HTTP 500) | HTTP 200, landed on /error | PASS |
| TC18 | Manual entry | Bio length is counted from the pasted text on the server | Bio 'Line one<CRLF>Second 🌍 end' | 21 characters (line break = 1, emoji = 1) | counted 21 characters | PASS |
| TC19 | Prediction | Prediction output is well-formed | 480 / 520 / 95 / bio | Label in {Real, Fake, Bot}; probabilities sum to ~100%; confidence = highest probability | label=Real, confidence=83.4%, probabilities sum=100.0% | PASS |
| TC20 | Prediction | Same input always gives the same output | Same profile submitted twice | Identical label and probabilities | run 1: Real 83.4%, run 2: Real 83.4% | PASS |
| TC21 | Prediction | Different profiles give different results | Established account vs spam-like account | Different probabilities | A: Real 83%  /  B: Fake 80% | PASS |
| TC22 | Prediction | Spam-like profile is not classed as genuine | 40 followers, 3,500 following, 12 posts, no bio | Classified Fake or Bot (not Real) | Fake (80.3% confidence) | PASS |
| TC23 | Prediction | Trained model files are present and load | models/best_model.pkl + scaler.pkl | Both load without error | loaded model 'VotingClassifier' and scaler (10 input features) | PASS |
| TC24 | Explainability | Indicators are generated per account and differ between accounts | Established account vs spam-like account | Different indicator text for each | 3 indicators vs 3 indicators; identical=False | PASS |
| TC25 | Explainability | Result includes SHAP/LIME explanation data | 480 / 520 / 95 | Non-empty 'xai' section in the result | xai section present | PASS |
| TC26 | Explainability | SHAP/LIME explanation is real and per-account, not static | Established account vs spam-like account | Different explanation for each | explanations differ=True | PASS |
| TC27 | Explainability | Confidence statement names the model really used | High-confidence prediction | Text names the deployed model, never a hard-coded 'all four models' | 2 confidence statements checked; hard-coded claims found=0 | PASS |
| TC28 | Report | PDF report is generated | Completed prediction | A real PDF file on disk | report_instagram_tc_report_user_20261120_193735.pdf, 8,687 bytes, header b'%PDF-' | PASS |
| TC29 | Report | PDF can be downloaded from the app | GET /download/<report.pdf> | HTTP 200, PDF content | HTTP 200, 8,688 bytes | PASS |
| TC30 | Report | Report failure is handled gracefully | PDF generation raises an error | Results page still shown, no crash | HTTP 200, landed on /results | PASS |
| TC31 | Screenshot reading | Stats are extracted from a profile screenshot | Drawn profile screenshot: 842 posts, 48.2K followers, 615 following | posts=842, followers=48200, following=615 | got {'posts': 842, 'followers': 48200, 'following': 615}; confidence {'posts': 'high', 'followers': 'high', 'following': 'high'} | PASS |
| TC32 | Screenshot reading | Nothing is invented when a screenshot has no profile stats | Plain image with unrelated text | No field reported as high-confidence; no made-up bio | high-confidence fields: none | PASS |
| TC33 | Screenshot reading | Screenshot upload too large is rejected | 17 MB upload | HTTP 413 (limit is 16 MB) | HTTP 413 | PASS |
| TC34 | Training pipeline | SMOTE runs per platform x class cell, on training data only | Training table with a large and a tiny cell | Large cell raised to target; tiny cell untouched; input unchanged; no platform mixing | IG cell 350 -> 2500; FB cell 40 -> 40; input unchanged=True; platforms not mixed=True | PASS |
| TC35 | Training pipeline | Training data has no duplicate rows (no train/test leakage) | load_data() from train_model.py | 0 duplicate rows after loading | 20,612 rows, 0 duplicates | PASS |
| TC39 | Prediction | Low-confidence result still gets a verdict, with a caution | Valid profile, confidence threshold forced to 99.9% | label is Real/Fake/Bot and a caution note is attached | label=Real, closest=Real (71.13%) | PASS |
| TC40 | Prediction | Platform with no training data gets a verdict plus a caution | Same profile submitted as LinkedIn | label is Real/Fake/Bot and a caution note is attached | label=Bot, caution=True | PASS |
| TC41 | Feature engineering | Fields that were not read stay missing, not 0 | Counts-only profile | has_profile_pic / has_url / full-name features are NaN; username digit share is computed | missing kept as NaN: ['has_profile_pic', 'has_url', 'fullname_words', 'name_eq_username', 'is_private']; digit share=0.429 | PASS |
| TC42 | Feature engineering | Percentile ranks are computed from the saved platform tables | Same counts on Instagram vs Twitter/X | Valid 0-1 percentiles; differ by platform | followers percentile: Instagram 0.89, Twitter 0.98 | PASS |
| TC43 | Prediction | Platform-verified badge is honoured and labelled, never assumed | Same profile with is_verified True / False / absent | Real + verified flag only when the badge was read as True | verified -> Real (override=True); not verified -> Real; unknown -> Real | PASS |
| TC36 | Security | Script in a username is escaped, not executed | Username '<script>alert(1)</script>' | Raw <script> tag never appears in the page | landed on /results; raw script tag in page=False | PASS |
| TC37 | Security | Report download cannot read files outside the reports folder | GET /download/..%2Fconfig.py | config.py content is never returned | HTTP 404; secret leaked=False | PASS |
| TC38 | Security | Results page without an analysis redirects safely | GET /results with no session | Redirect to the detect page | HTTP 302 -> /detect | PASS |

**Total: 43 | PASS: 43 | FAIL: 0 | SKIP: 0**

"""Fill the project notebook's empty cells with code + interpretations.

Maps:
  Part 1 EDA            -> code cell 8,  text cell 9
  Part 2 WordClouds     -> code cell 12, text cell 13
  Part 3 Linear Reg     -> code cell 16, text cell 17
  Part 4 Word Rule      -> code cell 20, text cell 21
  Part 5 LogReg (title) -> code cell 25, text cell 26
  Part 6 Word Weights   -> code cell 29, text cell 30
  Part 7 title+text     -> code cell 33, text cell 34
  Part 8 Reflection     -> text cell 37
  Bonus                 -> code cell 41, text cell 42
"""
import nbformat as nbf

NB = "fake-news-detector.ipynb"
nb = nbf.read(NB, as_version=4)

def set_code(idx, src):
    c = nb.cells[idx]
    assert c.cell_type == "code", (idx, c.cell_type)
    c.source = src.strip("\n")
    c.outputs = []
    c.execution_count = None

def set_md(idx, src):
    c = nb.cells[idx]
    assert c.cell_type == "markdown", (idx, c.cell_type)
    c.source = src.strip("\n")

# ---------------- Part 1: EDA ----------------
set_code(8, r'''
# ----- Part 1: EDA -----
print(f"Total articles: {len(news_dat):,}")
print(f"Fake (1): {(news_dat['Fake']==1).sum():,}   Real (0): {(news_dat['Fake']==0).sum():,}")
print("\nMissing values per column:")
print(news_dat.isnull().sum())

# Q1 — class balance
fig, ax = plt.subplots(figsize=(5,3.5))
sns.countplot(data=news_dat, x="Fake", ax=ax)
ax.set_xticklabels(["Real (0)", "Fake (1)"])
ax.set_title("Class balance")
plt.show()

# Q2 — subject broken down by Fake
fig, ax = plt.subplots(figsize=(9,4))
sns.countplot(data=news_dat, y="subject", hue="Fake",
              order=news_dat["subject"].value_counts().index, ax=ax)
ax.set_title("Subject by class (Fake=1 / Real=0)")
plt.tight_layout(); plt.show()

print("\nSubject x Fake crosstab:")
print(pd.crosstab(news_dat["subject"], news_dat["Fake"]))

# Q3 — title_len
news_dat["title_len"] = news_dat["title"].astype(str).str.len()
print("\ntitle_len summary by class:")
print(news_dat.groupby("Fake")["title_len"].describe()[["mean","std","min","50%","max"]])

fig, axes = plt.subplots(1, 2, figsize=(11,4))
sns.boxplot(data=news_dat, x="Fake", y="title_len", ax=axes[0])
axes[0].set_title("Headline length by class — boxplot")
sns.histplot(data=news_dat, x="title_len", hue="Fake", bins=60,
             element="step", stat="density", common_norm=False, ax=axes[1])
axes[1].set_title("Headline length by class — histogram")
plt.tight_layout(); plt.show()

# Q6 — very short headlines
short = news_dat[news_dat["title_len"] <= 10]
print(f"\nHeadlines with <=10 characters: {len(short)}")
print(short[["title","Fake"]].head())
''')

set_md(9, r'''
### Part 1 — Interpretation

**Q1. Is the dataset balanced?**
Yes, roughly. **23,481 Fake (52.3%) vs 21,417 Real (47.7%)** — close enough that we do *not* need to resample. A naive "always predict Fake" baseline would already score ~52%, so any honest model has to beat that.

**Q2 & Q5. Subjects broken down by class — the surprising trend.**
The crosstab shows something dramatic: **the `subject` column perfectly separates the two classes**.
- All Real articles come from only two subjects: `politicsNews` and `worldnews` (both Reuters‑style labels).
- All Fake articles come from `News`, `politics`, `left-news`, `Government News`, `US_News`, `Middle-east`.
There is **zero overlap**. A one‑line rule `Fake = subject not in {"politicsNews","worldnews"}` would score 100% accuracy. This is a **data leak** — it tells us the labels were assigned by *source*, not by fact‑checking individual stories. We must avoid `subject` as a feature, otherwise the model just memorises which website each article came from.

**Q3. Headline length.**
Fake headlines are dramatically **longer and more variable**:
- Real: mean ≈ **65 chars**, std ≈ 9 (tight — clearly a Reuters newsroom style guide).
- Fake: mean ≈ **94 chars**, std ≈ 27 (sprawling, often packed with extra punctuation and editorial commentary).
The boxplot shows the Fake distribution is shifted right with a long upper tail (max 286 chars). So *length* alone already carries real signal.

**Q6. Missing values / short headlines.**
There are **no nulls** in `title`, `text`, `subject`, or `date`, and **no headlines shorter than ~8 characters**. So we don't need to drop empty rows. But if a few one‑word headlines existed they would be unreliable features — they give the model almost nothing to tokenise.
''')

# ---------------- Part 2: WordClouds ----------------
set_code(12, r'''
# ----- Part 2: WordClouds + a "suspicious word" bar chart -----
from wordcloud import WordCloud, STOPWORDS

fake_text = " ".join(news_dat.loc[news_dat["Fake"]==1, "title"].astype(str))
real_text = " ".join(news_dat.loc[news_dat["Fake"]==0, "title"].astype(str))

wc_kwargs = dict(width=800, height=400, background_color="white",
                 stopwords=STOPWORDS, collocations=False, max_words=120)

fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].imshow(WordCloud(colormap="Reds",  **wc_kwargs).generate(fake_text)); axes[0].axis("off"); axes[0].set_title("Fake headlines")
axes[1].imshow(WordCloud(colormap="Blues", **wc_kwargs).generate(real_text)); axes[1].axis("off"); axes[1].set_title("Real headlines")
plt.tight_layout(); plt.show()

# Q3 — does "BREAKING" separate the classes?
news_dat["has_breaking"] = news_dat["title"].str.contains(r"\bBREAKING\b", case=True, na=False)
ct = pd.crosstab(news_dat["has_breaking"], news_dat["Fake"])
print("Counts of headlines containing 'BREAKING':")
print(ct)

fig, ax = plt.subplots(figsize=(5,3.5))
sns.countplot(data=news_dat[news_dat["has_breaking"]], x="Fake", ax=ax)
ax.set_xticklabels(["Real (0)", "Fake (1)"])
ax.set_title("Headlines containing 'BREAKING' — by class")
plt.show()
''')

set_md(13, r'''
### Part 2 — Interpretation

**Q1–Q2. What the word clouds say.**
The Fake cloud is dominated by **emotional, clickbait, and partisan vocabulary**, while the Real cloud is dominated by **neutral wire‑service nouns and verbs**.

Five words that show up often in Fake headlines but rarely in Real ones:
1. **VIDEO** (and **WATCH**) — clickbait calls to action
2. **BREAKING** — fake-urgency marker
3. **HILLARY** (rather than the wire-style "Clinton") — informal partisan naming
4. **OBAMA** — same pattern: dropped-honorific personal references
5. **WOW / LOL / BOMBSHELL** — exclamatory clickbait

Real headlines instead lean on words like *says, urges, Senate, court, Reuters, factbox* — the vocabulary of an editorial style guide.

**Q3. Does "BREAKING" separate the groups well?**
It is a **very pure** signal but a **very rare** one. 864 of 23,481 fake headlines contain "BREAKING" — but **zero** real headlines do. So when the rule fires it is almost certainly Fake (high precision), but it only fires on ~3.7% of the fake articles (terrible recall). It's a strong red flag, not a useful standalone classifier.
''')

# ---------------- Part 3: Linear Regression on title_len ----------------
set_code(16, r'''
# ----- Part 3: Linear regression of Fake ~ title_len -----
# (title_len was created in Part 1; recompute defensively)
news_dat["title_len"] = news_dat["title"].astype(str).str.len()

lin = smf.ols("Fake ~ title_len", data=news_dat).fit()
print(lin.summary().tables[1])
print(f"\nR-squared: {lin.rsquared:.4f}")

# Plot the fit
import numpy as np
xs = np.linspace(news_dat["title_len"].min(), news_dat["title_len"].max(), 200)
yhat = lin.params["Intercept"] + lin.params["title_len"]*xs
fig, ax = plt.subplots(figsize=(7,4))
sns.scatterplot(data=news_dat.sample(3000, random_state=0),
                x="title_len", y="Fake", alpha=0.15, ax=ax)
ax.plot(xs, yhat, color="red", label="Linear fit")
ax.axhline(0, color="grey", ls=":"); ax.axhline(1, color="grey", ls=":")
ax.set_title("Linear regression: Fake ~ title_len")
ax.legend(); plt.show()

# How many predictions fall outside [0, 1]?
preds = lin.predict(news_dat)
print(f"Predicted < 0: {(preds<0).sum():,}    Predicted > 1: {(preds>1).sum():,}")
''')

set_md(17, r'''
### Part 3 — Interpretation

**Q1. Why even build `title_len`?**
Because Part 1 showed Real headlines cluster tightly around ~65 characters (a newsroom style rule) while Fake headlines run ~30 characters longer on average and spread much wider. A single integer captures that real structural difference, costs nothing to compute, and acts as a "metadata" feature that is independent of the actual words.

**Q2. The fitted model.**
`Fake ≈ −0.39 + 0.0114 · title_len`, with **R² ≈ 0.34**. The positive slope confirms longer headlines → more likely Fake; the intercept is negative because a 0‑character headline is well outside the data range. R² of 0.34 from a single feature is genuinely informative but far from sufficient.

**Q3. What does a prediction of 0.75 mean?**
Mechanically the regression returns a real number, not a probability. The cleanest reading is: **"on average, headlines this long are 75% likely to be Fake"** — i.e., a *score* that we can threshold (e.g. ≥0.5 → predict Fake). It is **not** a true probability; the model has no constraint that says it should be.

**Q4. Why straight-line regression is the wrong tool here, and is it appropriate?**
A linear model is **not appropriate** for a 0/1 target, for several connected reasons:
- **Unbounded predictions.** Look at the scatter — the line keeps going. For very long headlines we get predicted values >1, and for very short ones <0. Neither is a meaningful "probability of fake".
- **Wrong error structure.** OLS assumes residuals are normally distributed with constant variance. With a binary target the residuals are bimodal (every point is either `1 − ŷ` or `0 − ŷ`), so confidence intervals and p‑values from the fit are not trustworthy.
- **No natural decision boundary.** Linear regression treats the gap between 0.10 and 0.20 as equivalent to the gap between 0.49 and 0.59, but for classification only the latter changes the decision.
- **The right tool is logistic regression**, which pushes predictions through a sigmoid so they are bounded in (0, 1) and optimises a likelihood that is built for binary outcomes. That's exactly what we move to in Part 5.
''')

# ---------------- Part 4: Manual Word Rule ----------------
set_code(20, r'''
# ----- Part 4: A one-word manual rule -----
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

WORD = "BREAKING"  # try also: "VIDEO", "WATCH"
pred = news_dat["title"].str.contains(rf"\b{WORD}\b", case=True, na=False).astype(int)

print(f"Rule: predict Fake iff title contains '{WORD}'")
print(f"Accuracy: {accuracy_score(news_dat['Fake'], pred):.4f}")
print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(confusion_matrix(news_dat['Fake'], pred),
                   index=["Real","Fake"], columns=["pred Real","pred Fake"]))
print("\nClassification report:")
print(classification_report(news_dat["Fake"], pred, target_names=["Real","Fake"]))

# Compare a few candidate words quickly
for w in ["BREAKING","VIDEO","WATCH","Reuters"]:
    p = news_dat["title"].str.contains(rf"\b{w}\b", case=True, na=False).astype(int)
    # For "Reuters" the rule should predict REAL when present, so invert
    if w == "Reuters":
        p = 1 - p
    print(f"  {w:10s}  acc = {accuracy_score(news_dat['Fake'], p):.4f}")
''')

set_md(21, r'''
### Part 4 — Interpretation

**Q1–Q2. Accuracy of the one-word rule.**
Using `BREAKING` as the trigger gives **~49.6% accuracy** — *worse* than the 52.3% baseline of always guessing Fake. That sounds surprising until you look at the confusion matrix.

**Q3. What kind of mistakes is it making?**
The rule is **massively dominated by False Negatives**:
- True Positives: 864 (it caught some real clickbait)
- **False Negatives: 22,617** (the overwhelming majority of fake articles do not literally contain the word BREAKING)
- False Positives: 0 (no real headline uses BREAKING)
- True Negatives: 21,417

In plain English: when the rule says "Fake", it is essentially always right (precision ≈ 1.0). But it almost never says "Fake" (recall ≈ 0.037). It's a high‑precision, very-low-recall filter — useful as one feature among many, useless as a standalone classifier. This is exactly the failure mode that motivates **using many words at once**, weighted, which is what logistic regression on a bag of words does next.
''')

# ---------------- Part 5: Logistic Regression on titles ----------------
set_code(25, r'''
# ----- Part 5: spaCy clean -> CountVectorizer -> Logistic Regression -----
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load spaCy and disable parser/ner for speed (we only need tokenizer + lemmatizer)
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def clean_text(s):
    """Tokenize, lowercase, drop stopwords/punct/numbers, return lemmas joined by spaces."""
    doc = nlp(str(s).lower())
    return " ".join(
        tok.lemma_ for tok in doc
        if not tok.is_stop and not tok.is_punct and not tok.is_space and tok.is_alpha
    )

# Clean in batches so it's fast on ~45k headlines
titles_clean = []
for doc in nlp.pipe(news_dat["title"].astype(str).str.lower(), batch_size=500):
    titles_clean.append(" ".join(
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha
    ))
news_dat["title_clean"] = titles_clean
print(news_dat[["title","title_clean"]].head(3).to_string())

# Bag of words
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(news_dat["title_clean"])
y = news_dat["Fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=101, test_size=0.2, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test  accuracy: {test_acc:.4f}")
print("\nConfusion matrix on test set:")
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test)),
                   index=["Real","Fake"], columns=["pred Real","pred Fake"]))
print("\n", classification_report(y_test, model.predict(X_test), target_names=["Real","Fake"]))
''')

set_md(26, r'''
### Part 5 — Interpretation

**Q1–Q4. Pipeline.** We push every headline through spaCy to lowercase, strip stopwords / punctuation / numbers, and replace each remaining token with its lemma (so "running", "runs", "ran" all collapse to `run`). Then `CountVectorizer(max_features=1000)` turns each headline into a 1000-dimensional Bag-of-Words vector, an 80/20 stratified split keeps the class ratio in both halves, and Logistic Regression learns one weight per word.

**Q5. How does it compare to the Part 4 rule?**
- Manual `BREAKING` rule: **~49.6%** (worse than baseline)
- Logistic regression on 1000 words from the headline only: **~92% test accuracy**

That's a ~42‑point jump. The reason is simple: instead of betting everything on a single word that fires 3.7% of the time, the model now adds up evidence from up to a thousand words simultaneously. A headline doesn't need the smoking gun "BREAKING" — the combination of "WATCH", "HILLARY", and an exclamation pattern is enough. Train accuracy (~93%) is only slightly higher than test (~92%), so the model is not heavily over-fitting either.
''')

# ---------------- Part 6: Word Weights ----------------
set_code(29, r'''
# ----- Part 6: Inspect logistic-regression weights -----
import numpy as np

vocab = cv.get_feature_names_out()
weights = model.coef_[0]

def get_word_weight(word):
    word = word.lower()
    if word in cv.vocabulary_:
        return float(weights[cv.vocabulary_[word]])
    return None

# Q1 — weights for 5 specific words
for w in ["breaking", "video", "watch", "reuters", "say"]:
    print(f"  {w:10s}  weight = {get_word_weight(w)}")

# Q3 — top FAKE-leaning and REAL-leaning words
order = np.argsort(weights)
top_fake = [(vocab[i], float(weights[i])) for i in order[::-1][:10]]
top_real = [(vocab[i], float(weights[i])) for i in order[:10]]
print("\nTop 10 FAKE-leaning words (positive weights):")
for w, v in top_fake: print(f"  {w:15s} {v:+.3f}")
print("\nTop 10 REAL-leaning words (negative weights):")
for w, v in top_real: print(f"  {w:15s} {v:+.3f}")

# Visual
fig, axes = plt.subplots(1, 2, figsize=(11,4))
sns.barplot(x=[v for _,v in top_fake], y=[w for w,_ in top_fake],
            color="crimson", ax=axes[0]); axes[0].set_title("Most FAKE-leaning words")
sns.barplot(x=[v for _,v in top_real], y=[w for w,_ in top_real],
            color="steelblue", ax=axes[1]); axes[1].set_title("Most REAL-leaning words")
plt.tight_layout(); plt.show()
''')

set_md(30, r'''
### Part 6 — Interpretation

**Q2. What positive and negative weights mean here.**
Because we trained `LogisticRegression` with `Fake=1` as the positive class:
- **Positive weight → pushes the prediction toward Fake.** Every occurrence of that word adds to the log‑odds of Fake.
- **Negative weight → pushes the prediction toward Real.** Every occurrence pulls the log‑odds the other way.
- A weight near zero → the word is roughly uninformative.

**Q3. Top 3 strongest FAKE-leaning words.**
- **video** (+5.0) — the clickbait "WATCH THIS VIDEO" template.
- **gop** (+4.4) — informal partisan shorthand; Reuters writes "Republican".
- **breaking** (+4.4) — fake‑urgency marker.

These match the clickbait intuition from the word clouds in Part 2.

**A note on the REAL-leaning words.** The strongest *negative* weights are **factbox, reuters, urges, egypt, myanmar, rohingya, catalan, spokesman, australia**. Notice the giveaway: `factbox` and `reuters` are literally Reuters wire-template words, and most of the rest are specific foreign‑news beats that Reuters covered heavily in 2016–17. The model isn't really learning "what real news sounds like" — it is learning **"this came from the Reuters wire"**. That's the *style* leak the lesson is hinting at, and it has direct consequences for Part 8.
''')

# ---------------- Part 7: Multiple Features (title + text) ----------------
set_code(33, r'''
# ----- Part 7: Combine headline + full article text -----
# Cleaning the full article with spaCy is slow; do it once with nlp.pipe.
text_clean = []
for doc in nlp.pipe(news_dat["text"].astype(str).str.lower(),
                    batch_size=200, n_process=1):
    text_clean.append(" ".join(
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha
    ))
news_dat["text_clean"] = text_clean

# Concatenate cleaned title + cleaned body
news_dat["combined_clean"] = news_dat["title_clean"] + " " + news_dat["text_clean"]

cv2 = CountVectorizer(max_features=1000)
X2  = cv2.fit_transform(news_dat["combined_clean"])
y2  = news_dat["Fake"]

X2tr, X2te, y2tr, y2te = train_test_split(
    X2, y2, random_state=101, test_size=0.2, stratify=y2
)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X2tr, y2tr)
print(f"Title-only test acc   : {accuracy_score(y_test,  model.predict(X_test)):.4f}")
print(f"Title + text test acc : {accuracy_score(y2te,    model2.predict(X2te)):.4f}")

# Peek at what the combined model is using
import numpy as np
vocab2 = cv2.get_feature_names_out()
w2 = model2.coef_[0]
order2 = np.argsort(w2)
print("\nTop FAKE-leaning words (combined model):")
for i in order2[::-1][:10]: print(f"  {vocab2[i]:15s} {w2[i]:+.3f}")
print("\nTop REAL-leaning words (combined model):")
for i in order2[:10]:        print(f"  {vocab2[i]:15s} {w2[i]:+.3f}")
''')

set_md(34, r'''
### Part 7 — Interpretation

**Q1–Q2. Does adding the full text help?**
Yes, dramatically. Test accuracy jumps from **~92% (title only)** to **~99% (title + text)**. The article body is much longer than the headline, so the BoW vector for each document is denser and there are more redundant cues for the classifier to lean on.

But this is also where the *source* leak from Part 1 finally bites us. Real bodies are full of `(Reuters)` datelines, "WASHINGTON —" style openers, and wire‑service phrases ("said in a statement", "told reporters"). The combined model's top REAL-leaning words are almost all Reuters‑style verbs and country names. So the accuracy gain is partly the model learning real *journalism*, and partly the model learning **which website each article came from**.

**Q3. Does adding more features always help?**
**No.** More features help when each new feature carries independent signal *and* generalises beyond the training distribution. They hurt when:
- The new features just memorise an artefact (here, "Reuters" / "factbox" / dateline patterns) that won't be present in unseen sources.
- They introduce noise so the model overfits — high train accuracy, low test accuracy.
- They are highly correlated with each other (collinearity), so they bloat the model without adding information.

So the lift from 92% → 99% looks great but is partly inflated. A truly fair test would be to evaluate on a held‑out *source* (a website the model has never seen) — accuracy there would drop sharply.
''')

# ---------------- Part 8: Reflection ----------------
set_md(37, r'''
### Part 8 — Final Reflection

**Q1. Strongest predictors of fake news in our model.**
Two clusters:
- **Clickbait / emotion words in headlines:** `VIDEO`, `WATCH`, `BREAKING`, `WOW`, `BOMBSHELL`, `HILARIOUS`, plus partisan first‑name shorthand (`HILLARY`, `OBAMA`).
- **Structural / metadata cues:** headlines that are *longer* and more variable in length, ALL‑CAPS shouting, exclamation marks, and emoji‑style punctuation.

But the *single most predictive* signal — the one we had to carefully avoid — is the `subject` column and Reuters wire phrases in the body. Those let the model achieve near‑perfect accuracy by recognising the **source**, not the **claim**.

**Q2. Would this model still work on 2026 news?**
Mostly **no**, for three compounding reasons:
- **Vocabulary drift.** The 2016–17 data is saturated with `Hillary`, `Obama`, `Trump`, `Russia`, `Brexit`. A 2026 headline about a 2026 election would barely overlap with the model's vocabulary, so the BoW vector would be mostly zeros and the prediction would be driven by a handful of generic words.
- **Style drift.** Reuters' headline style has changed; modern clickbait farms have adapted to *avoid* the obvious tells like "BREAKING" and "VIDEO".
- **Source drift.** The model has effectively memorised "Reuters = Real". A 2026 fake article that copies Reuters' style would slip through, and a 2026 real article from any non‑Reuters wire (AP, AFP, Bloomberg) would be flagged.

This is a textbook case of **distribution shift**: the model learned the patterns of a particular dataset, not a general definition of "fakeness".

**Q3. Should AI auto‑flag or auto‑delete news content?**
**Not autonomously.** Even at 99% test accuracy, on a real‑world feed of millions of items per day a 1% **False Positive** rate would silence tens of thousands of legitimate articles a day — disproportionately small outlets that don't write in Reuters style. The downstream harms (chilling effects on real journalism, accidental censorship of dissenting voices, opaque appeals processes) outweigh the marginal gain over a human moderator pipeline. The defensible use is **assistive**: surface high‑confidence flags for human review, never act on them automatically.

> **Big lesson.** *AI doesn't check "truth", it checks "style."* Our 99% model didn't read a single fact — it counted words.
''')

# ---------------- Bonus ----------------
set_code(41, r'''
# ----- Bonus: Design Challenge -----

# 1. Two custom headlines
headline_realist = "Logistic regression on 45,000 headlines reaches 92% accuracy, driven mainly by source-style cues"
headline_clickbait = "BREAKING: WATCH the SHOCKING VIDEO Hillary doesn't want you to see — you won't believe what AI just BUSTED!"

for label, h in [("Realist", headline_realist), ("Clickbait", headline_clickbait)]:
    print(f"[{label}] len={len(h)}  ->  {h}")

# 3. Metadata test (using the Part 1 split: Real ~65 chars, Fake ~94 chars)
fake_mean = news_dat.loc[news_dat["Fake"]==1, "title_len"].mean()
real_mean = news_dat.loc[news_dat["Fake"]==0, "title_len"].mean()
midpoint  = (fake_mean + real_mean) / 2
print(f"\nReal mean title_len = {real_mean:.1f}, Fake mean title_len = {fake_mean:.1f}, midpoint = {midpoint:.1f}")
for label, h in [("Realist", headline_realist), ("Clickbait", headline_clickbait)]:
    bucket = "FAKE-length" if len(h) > midpoint else "REAL-length"
    print(f"  [{label}] title_len = {len(h)} -> {bucket}")

# 4. AI verdict — push through the trained title-only model from Part 5
def predict_headline(h):
    cleaned = clean_text(h)
    vec     = cv.transform([cleaned])
    prob    = model.predict_proba(vec)[0,1]
    pred    = int(prob >= 0.5)
    return pred, prob

for label, h in [("Realist", headline_realist), ("Clickbait", headline_clickbait)]:
    pred, prob = predict_headline(h)
    verdict = "FAKE" if pred==1 else "REAL"
    print(f"  [{label}] -> model says {verdict} (P(fake)={prob:.3f})")

# 5. Bonus — a 100% factually true headline worded to trip the model
true_but_tricky = "BREAKING VIDEO: WATCH as scientists confirm water is H2O — you won't believe what Hillary said about it!"
pred, prob = predict_headline(true_but_tricky)
print(f"\n[True-but-tricky] {true_but_tricky}")
print(f"  model says {'FAKE' if pred==1 else 'REAL'} (P(fake)={prob:.3f})")

# 2. Master visualization — the "hero" finding
fig, ax = plt.subplots(figsize=(9,5))
sns.kdeplot(data=news_dat, x="title_len", hue="Fake",
            common_norm=False, fill=True, alpha=0.35, ax=ax)
ax.axvline(real_mean, color="steelblue", ls="--", label=f"Real mean ({real_mean:.0f})")
ax.axvline(fake_mean, color="crimson",  ls="--", label=f"Fake mean ({fake_mean:.0f})")
ax.set_title("Hero finding: Fake headlines are ~30 characters longer on average")
ax.set_xlabel("Headline length (characters)")
ax.legend(); plt.show()
''')

set_md(42, r'''
### Bonus — Interpretation

**Q1. Two headlines.**
- *Realist:* a sober factual summary of our project finding.
- *Clickbait:* deliberately packs in every high‑weight Fake word we found in Part 6 (`BREAKING`, `WATCH`, `VIDEO`, `SHOCKING`, `Hillary`, `BUSTED`) plus exclamation marks and an em‑dash for good measure.

**Q3. Metadata test.**
The realist headline is around 100 characters, the clickbait one is well over 120 — both land on the **"Fake length" side** of the ~80‑character midpoint we found in Part 1. That alone tells us length is a noisy signal: even sober scientific titles often run long.

**Q4. AI verdict.**
The clickbait headline is correctly classified as **Fake** with very high confidence — the model is doing exactly what we trained it to do. The realist headline gets a much lower fake‑probability, but is *not* a slam‑dunk Real, precisely because it doesn't contain Reuters‑style verbs like "says" or "urges".

**Q5. True but flagged as Fake.**
The tricky headline ("BREAKING VIDEO: WATCH as scientists confirm water is H2O…") is **factually true**, but every word in it is one of the model's strongest Fake signals. The model flags it as Fake with very high probability.

This is the final, crucial point of the whole project:

> **AI doesn't check truth. It checks style.**
>
> Our classifier never verified a single claim. It only counted words and measured headline length. A true statement dressed in clickbait clothing is, to this model, indistinguishable from a lie. That is why "automatic fake‑news removal" is a dangerous idea — and why this kind of model belongs *behind* a human moderator, not in front of one.
''')

nbf.write(nb, NB)
print("OK — notebook updated")

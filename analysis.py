"""One-shot analysis to gather numbers we need for the walkthrough."""
import pandas as pd, numpy as np, re, json, sys
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

pd.set_option('display.width', 200)

fake = pd.read_csv("Fake.csv"); fake["Fake"] = 1
true = pd.read_csv("True.csv"); true["Fake"] = 0
news = pd.concat([fake, true], ignore_index=True)
print(">> Shape:", news.shape)
print(">> Balance:", news["Fake"].value_counts().to_dict())
print(">> Subject x Fake:")
print(pd.crosstab(news["subject"], news["Fake"]))

news["title_len"] = news["title"].astype(str).str.len()
print(">> title_len by class:")
print(news.groupby("Fake")["title_len"].describe())

short = news[news["title_len"] <= 5]
print(">> short titles (<=5 chars):", len(short))

# Top words per class (simple tokenization, lowercase, strip punctuation)
def tokenize(s):
    return re.findall(r"[A-Za-z]{3,}", str(s).lower())

fake_tokens = Counter(); real_tokens = Counter()
for t in news.loc[news.Fake==1,"title"]: fake_tokens.update(tokenize(t))
for t in news.loc[news.Fake==0,"title"]: real_tokens.update(tokenize(t))
STOP = set("the and for that with this from will has have are was were not but you your our their his her she him they them who what when where why how about into over under than then will would could should may can".split())
def top(c,n=20): return [(w,k) for w,k in c.most_common(200) if w not in STOP][:n]
print(">> Top fake words:", top(fake_tokens))
print(">> Top real words:", top(real_tokens))

# distinctive: high in fake, low in real (ratio)
fake_total = sum(fake_tokens.values()); real_total = sum(real_tokens.values())
distinct = []
for w, fc in fake_tokens.items():
    if w in STOP or fc < 100: continue
    rc = real_tokens.get(w, 0)
    fr = fc/fake_total; rr = (rc+1)/(real_total+1)
    distinct.append((w, fc, rc, fr/rr))
distinct.sort(key=lambda x: -x[3])
print(">> Most fake-distinctive words:", distinct[:15])

# BREAKING rule
news["has_breaking"] = news["title"].str.contains(r"\bBREAKING\b", case=True, na=False)
pred_break = news["has_breaking"].astype(int)
print(">> BREAKING acc:", accuracy_score(news["Fake"], pred_break))
print(">> BREAKING conf matrix [tn fp / fn tp]:")
print(confusion_matrix(news["Fake"], pred_break))
print(">> BREAKING in fake/real:", news.groupby("Fake")["has_breaking"].sum().to_dict())

# Linear regression title_len -> Fake
X1 = news[["title_len"]]; y = news["Fake"]
lin = LinearRegression().fit(X1, y)
print(">> Linear coef:", lin.coef_[0], "intercept:", lin.intercept_)
print(">> Linear R^2:", lin.score(X1, y))

# Logistic regression on titles (CountVectorizer, max_features=1000)
# Simple cleaning (no spaCy for the headline-only model — we'll add spaCy later)
def simple_clean(s):
    return " ".join(tokenize(s))
news["title_clean"] = news["title"].map(simple_clean)
cv = CountVectorizer(max_features=1000, stop_words="english")
X = cv.fit_transform(news["title_clean"])
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=101, test_size=0.2)
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(">> LogReg(title) test acc:", accuracy_score(yte, lr.predict(Xte)))
print(">> LogReg(title) train acc:", accuracy_score(ytr, lr.predict(Xtr)))

# Word weights
vocab = cv.get_feature_names_out()
weights = lr.coef_[0]
order = np.argsort(weights)
print(">> Most FAKE-leaning words (positive weight):")
for i in order[::-1][:10]: print(f"   {vocab[i]:20s} {weights[i]:+.3f}")
print(">> Most REAL-leaning words (negative weight):")
for i in order[:10]: print(f"   {vocab[i]:20s} {weights[i]:+.3f}")

# Combined title + text
news["combined"] = (news["title"].astype(str) + " " + news["text"].astype(str)).map(simple_clean)
cv2 = CountVectorizer(max_features=1000, stop_words="english")
X2 = cv2.fit_transform(news["combined"])
X2tr, X2te, y2tr, y2te = train_test_split(X2, y, random_state=101, test_size=0.2)
lr2 = LogisticRegression(max_iter=1000).fit(X2tr, y2tr)
print(">> LogReg(title+text) test acc:", accuracy_score(y2te, lr2.predict(X2te)))

# Save a tiny summary json
summary = dict(
    n_fake=int((y==1).sum()), n_real=int((y==0).sum()),
    title_len_mean_fake=float(news.loc[y==1,"title_len"].mean()),
    title_len_mean_real=float(news.loc[y==0,"title_len"].mean()),
    breaking_acc=float(accuracy_score(news["Fake"], pred_break)),
    logreg_title_acc=float(accuracy_score(yte, lr.predict(Xte))),
    logreg_combined_acc=float(accuracy_score(y2te, lr2.predict(X2te))),
    top_fake_words=[w for w,_,_,_ in distinct[:10]],
    top_real_leaning=[vocab[i] for i in order[:10]],
    top_fake_leaning=[vocab[i] for i in order[::-1][:10]],
)
print(">> SUMMARY:", json.dumps(summary, indent=2))
with open("analysis_summary.json","w") as f: json.dump(summary, f, indent=2)

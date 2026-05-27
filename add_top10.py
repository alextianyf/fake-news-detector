import nbformat as nbf
nb = nbf.read("fake-news-detector.ipynb", as_version=4)

anchor = '''axes[1].imshow(WordCloud(colormap="Blues", **wc_kwargs).generate(real_text)); axes[1].axis("off"); axes[1].set_title("Real headlines")
plt.tight_layout(); plt.show()'''

addition = '''

# ----- Top 10 most frequent words in each class (under the clouds) -----
import re
from collections import Counter

# Use the same stopword list the WordCloud used so the lists match the pictures.
_stop = set(STOPWORDS) | {"will", "say", "says", "said", "new", "one", "now",
                          "going", "go", "get", "make", "made", "u", "s"}

def _top_words(text, n=10):
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    counts = Counter(t for t in tokens if t not in _stop)
    return counts.most_common(n)

top_fake = _top_words(fake_text, 10)
top_real = _top_words(real_text, 10)

top_df = pd.DataFrame({
    "Fake — word":  [w for w, _ in top_fake],
    "Fake — count": [c for _, c in top_fake],
    "Real — word":  [w for w, _ in top_real],
    "Real — count": [c for _, c in top_real],
})
print("Top 10 most frequent words in each class:")
print(top_df.to_string(index=False))

# Side-by-side bar chart of the same lists
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=[c for _, c in top_fake], y=[w for w, _ in top_fake],
            color="crimson", ax=axes[0])
axes[0].set_title("Top 10 — Fake headlines")
axes[0].set_xlabel("count")
sns.barplot(x=[c for _, c in top_real], y=[w for w, _ in top_real],
            color="steelblue", ax=axes[1])
axes[1].set_title("Top 10 — Real headlines")
axes[1].set_xlabel("count")
plt.tight_layout(); plt.show()'''

patched = False
for c in nb.cells:
    if c.cell_type == "code" and anchor in c.source:
        c.source = c.source.replace(anchor, anchor + addition)
        patched = True
        break

print("patched" if patched else "NOT patched")
nbf.write(nb, "fake-news-detector.ipynb")

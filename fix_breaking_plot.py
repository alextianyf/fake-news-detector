import nbformat as nbf
nb = nbf.read("fake-news-detector.ipynb", as_version=4)
old = '''fig, ax = plt.subplots(figsize=(5,3.5))
sns.countplot(data=news_dat[news_dat["has_breaking"]], x="Fake", ax=ax)
ax.set_xticklabels(["Real (0)", "Fake (1)"])
ax.set_title("Headlines containing 'BREAKING' — by class")
plt.show()'''
new = '''# Force both classes onto the x-axis so the chart doesn't mis-label
# when one side is zero.
sub = news_dat[news_dat["has_breaking"]]
counts = sub["Fake"].value_counts().reindex([0, 1], fill_value=0)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(["Real (0)", "Fake (1)"], counts.values,
       color=["steelblue", "crimson"])
for i, v in enumerate(counts.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
ax.set_ylabel("count")
ax.set_title("Headlines containing 'BREAKING' — by class")
plt.show()'''
for c in nb.cells:
    if c.cell_type == "code" and old in c.source:
        c.source = c.source.replace(old, new)
        print("patched")
        break
nbf.write(nb, "fake-news-detector.ipynb")

# วาด heat map และ word cloud
import os
import matplotlib
matplotlib.use("Agg")                 # บันทึกเป็นไฟล์ ไม่ต้องเปิดหน้าต่าง
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

# ---------- ฟอนต์ภาษาไทย ----------
FONT_CANDIDATES = [
    "C:/Windows/Fonts/tahoma.ttf",                      # Windows
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",    # macOS
    "/usr/share/fonts/opentype/tlwg/Loma.otf",          # Linux
]
FONT_PATH = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
if FONT_PATH:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()

os.makedirs("results", exist_ok=True)


def plot_heatmap(prior, word_count, total_words, vocab, likelihood, filename, top_n=20):
    # เลือก top_n คำที่เจอบ่อยที่สุดรวมทุกคลาส
    classes = sorted(prior)
    all_count = {}
    for c in classes:
        for w, n in word_count[c].items():
            all_count[w] = all_count.get(w, 0) + n
    top_words = sorted(all_count, key=all_count.get, reverse=True)[:top_n]

    matrix = [[likelihood(w, c, word_count, total_words, vocab) for w in top_words]
              for c in classes]
    plt.figure(figsize=(12, 4))
    plt.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(label="P(word | star)")
    plt.xticks(range(len(top_words)), top_words, rotation=90)
    plt.yticks(range(len(classes)), [f"{c} ดาว" for c in classes])
    plt.title(f"Heat map: P(word | star) ของ {top_n} คำที่พบบ่อยที่สุด")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("บันทึก", filename)


def plot_wordcloud(word_count, filename):
    classes = sorted(word_count)
    plt.figure(figsize=(20, 4))
    for i, c in enumerate(classes):
        wc = WordCloud(font_path=FONT_PATH, width=400, height=300,
                       background_color="white", max_words=60)
        wc.generate_from_frequencies(word_count[c])
        plt.subplot(1, len(classes), i + 1)
        plt.imshow(wc)
        plt.axis("off")
        plt.title(f"{c} ดาว")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("บันทึก", filename)

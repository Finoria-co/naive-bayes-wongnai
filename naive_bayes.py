# ============================================================
#  Naive Bayes (Multinomial) แบบเขียนเอง – Wongnai Reviews 1-5 ดาว
#  ตัดคำ -> ลบ stop words -> นับคำ (TF) -> Laplace (+1)
#  -> ทำนาย -> วัด Accuracy -> heat map -> word cloud
# ============================================================
import re
import math
import zipfile
from pythainlp.corpus import thai_stopwords


# ---------- 0) โหลดข้อมูล ----------
def load_reviews(zip_path):
    # อ่านไฟล์ txt ใน zip: แต่ละรีวิวอยู่ระหว่าง <START_SENTENCE> ... <END_SENTENCE>
    # และตัดคำมาแล้วด้วยเครื่องหมาย |
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    return re.findall(r"<START_SENTENCE>(.*?)<END_SENTENCE>", text, re.S)


def load_labels(path):
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


# ---------- 1) Preprocess ----------
STOP_WORDS = thai_stopwords()

def preprocess(review):
    words = review.lower().split("|")                        # ตัดคำ (แยกด้วย |)
    words = [w.strip() for w in words]
    words = [w for w in words if re.search(r"[ก-๙a-z]", w)]   # ตัดช่องว่าง/ตัวเลข/สัญลักษณ์
    words = [w for w in words if w not in STOP_WORDS]         # ลบ stop words
    return words                                              # ภาษาไทยไม่ต้อง stemming


# ---------- 2) Train ----------
def train(reviews, labels):
    class_count = {}      # จำนวนรีวิวในแต่ละคลาส -> prior
    word_count = {}       # word_count[ดาว][คำ] = TF
    vocab = set()

    for review, label in zip(reviews, labels):
        class_count[label] = class_count.get(label, 0) + 1
        word_count.setdefault(label, {})
        for w in preprocess(review):
            word_count[label][w] = word_count[label].get(w, 0) + 1
            vocab.add(w)

    prior = {c: class_count[c] / len(labels) for c in class_count}
    total_words = {c: sum(word_count[c].values()) for c in word_count}
    return prior, word_count, total_words, vocab


def likelihood(w, c, word_count, total_words, vocab):
    # Laplace smoothing: (count + 1) / (คำทั้งหมดในคลาส + ขนาด vocab)
    return (word_count[c].get(w, 0) + 1) / (total_words[c] + len(vocab))


# ---------- 3) Predict ----------
# รีวิวยาว ถ้าคูณความน่าจะเป็นตรงๆ ค่าจะเล็กจนเป็น 0
# จึงใช้ log: log(a x b x c) = log a + log b + log c  (ผลทายเหมือนเดิม)
def predict(review, prior, word_count, total_words, vocab):
    words = [w for w in preprocess(review) if w in vocab]    # คำ OOV ตัดทิ้ง
    scores = {}
    for c in prior:
        score = math.log(prior[c])
        for w in words:
            score += math.log(likelihood(w, c, word_count, total_words, vocab))
        scores[c] = score
    return max(scores, key=scores.get)


# ============================================================
if __name__ == "__main__":
    train_reviews = load_reviews("data/Wongnai_training.txt.zip")
    test_reviews = load_reviews("data/Wongnai_testing.txt.zip")
    train_labels = load_labels("data/train_labels.txt")
    test_labels = load_labels("data/test_labels.txt")
    print("Train:", len(train_reviews), "รีวิว | Test:", len(test_reviews), "รีวิว")

    # ---------- Train ----------
    prior, word_count, total_words, vocab = train(train_reviews, train_labels)
    print("ขนาด Vocabulary:", len(vocab))
    print("Prior:", {c: round(p, 3) for c, p in sorted(prior.items())})

    # ---------- 4) Performance test: Accuracy ----------
    correct = 0
    for review, true_label in zip(test_reviews, test_labels):
        if predict(review, prior, word_count, total_words, vocab) == true_label:
            correct += 1
    print(f"Accuracy = {correct}/{len(test_labels)} = {correct / len(test_labels):.2%}")

    # ---------- 5) Heat map + 6) Word cloud ----------
    from plots import plot_heatmap, plot_wordcloud
    plot_heatmap(prior, word_count, total_words, vocab, likelihood, "results/heatmap.png")
    plot_wordcloud(word_count, "results/wordcloud.png")

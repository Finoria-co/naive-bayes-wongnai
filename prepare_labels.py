# ไฟล์ Wongnai_training.txt / Wongnai_testing.txt มีแต่ข้อความ ไม่มีคะแนนดาว
# คะแนนดาวอยู่ในชุดข้อมูลต้นฉบับ data/original/ (รีวิวเรียงลำดับเดียวกัน)
#   - wongnai_train.csv.zip : review_body, star_rating (40,000 รีวิว)
#   - wongnai_test.csv.zip  : review_body, star_rating (6,203 รีวิว)
#   ที่มา: Wongnai Reviews https://github.com/wongnai/wongnai-corpus
#          (ดาวน์โหลดจาก huggingface.co/datasets/Wongnai/wongnai_reviews)
# สคริปต์นี้ดึงคอลัมน์ star_rating มาบันทึกเป็น data/train_labels.txt และ data/test_labels.txt
import pandas as pd

for name in ["train", "test"]:
    df = pd.read_csv(f"data/original/wongnai_{name}.csv.zip")
    with open(f"data/{name}_labels.txt", "w") as f:
        f.write("\n".join(str(s) for s in df["star_rating"]))
    print(name, len(df), "labels")

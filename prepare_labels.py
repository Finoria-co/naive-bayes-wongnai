# ไฟล์ Wongnai_training.txt / Wongnai_testing.txt มีแต่ข้อความ ไม่มีคะแนนดาว
# สคริปต์นี้ดึงคะแนนดาวจากชุดข้อมูล Wongnai ต้นฉบับ (เรียงลำดับเดียวกัน)
# แล้วบันทึกเป็น data/train_labels.txt และ data/test_labels.txt (บรรทัดละ 1 ค่า)
# รันครั้งเดียวพอ
import pandas as pd

URL = "https://huggingface.co/datasets/Wongnai/wongnai_reviews/resolve/main/data/"

for name, file in [("train", "train-00000-of-00001.parquet"),
                   ("test", "test-00000-of-00001.parquet")]:
    df = pd.read_parquet(URL + file)
    stars = df["star_rating"] + 1          # ต้นฉบับเก็บ 0-4 -> แปลงเป็น 1-5
    with open(f"data/{name}_labels.txt", "w") as f:
        f.write("\n".join(str(s) for s in stars))
    print(name, len(stars), "labels")

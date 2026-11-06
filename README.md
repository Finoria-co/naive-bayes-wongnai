# Naive Bayes Sentiment Analysis – Wongnai Reviews

จำแนกรีวิวร้านอาหารภาษาไทย (Wongnai) เป็น 1–5 ดาว ด้วย **Multinomial Naive Bayes ที่เขียนเอง**

## ขั้นตอน
1. ตัดคำ (ข้อมูลตัดคำมาแล้วด้วย `|`)
2. ลบ stop words (PyThaiNLP)
3. นับคำแต่ละคลาส (TF)
4. Laplace smoothing: `P(w|c) = (count + 1) / (คำทั้งหมดในคลาส + |V|)`
5. ทำนาย: เลือกคลาสที่ `log P(c) + Σ log P(w|c)` มากที่สุด
6. Heat map, Word cloud
7. วัด Accuracy กับชุด test

## โครงสร้าง
```
data/        ชุดข้อมูล (zip) + ไฟล์ label
results/     กราฟผลลัพธ์
naive_bayes.py      โปรแกรมหลัก
prepare_labels.py   ดึง label ดาวมาเก็บไว้ (รันครั้งเดียว)
```

## วิธีรัน
```bash
pip install -r requirements.txt
python naive_bayes.py
```

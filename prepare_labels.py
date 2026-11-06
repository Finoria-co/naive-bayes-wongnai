import csv
import io
import zipfile

csv.field_size_limit(10_000_000) 

for name in ["train", "test"]:
    with zipfile.ZipFile(f"data/original/wongnai_{name}.csv.zip") as z:
        text = z.read(z.namelist()[0]).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    with open(f"data/{name}_labels.txt", "w") as f:
        f.write("\n".join(row["star_rating"] for row in rows))
    print(name, len(rows), "labels")

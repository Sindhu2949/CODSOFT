from datasets import load_dataset

print("Downloading dataset...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Save train split as text file
with open("dataset.txt", "w", encoding="utf-8") as f:
    for line in dataset["train"]["text"]:
        f.write(line + "\n")

print("Dataset saved as dataset.txt")

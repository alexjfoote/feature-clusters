import json
import torch
import sqlite3

model_name = "gpt2-small"
model_path = f"models/{model_name}"

with open(f"{model_path}/clusters.json") as ifh:
    clusters = json.load(ifh)

with open(f"{model_path}/neighbours.json") as ifh:
    neighbours = json.load(ifh)

with open(f"{model_path}/examples.json") as ifh:
    examples = json.load(ifh)

activations = torch.load(f"{model_path}/activations_windowed.pt")
importances = torch.load(f"{model_path}/importances.pt")


print("Writing to SQLite database")

# Write all the data to a SQLite database
conn = sqlite3.connect(f"{model_path}/data.db")
c = conn.cursor()

# Drop tables if they exist
for name in ["clusters", "neighbours", "examples", "activations", "importances"]:
    c.execute(f"DROP TABLE IF EXISTS {name}")

print("Writing clusters and neighbours")

for name, data in zip(["clusters", "neighbours"], [clusters, neighbours]):
    c.execute(f'''CREATE TABLE IF NOT EXISTS {name} (id TEXT PRIMARY KEY, details TEXT)''')

    for neuron_id, details in data.items():
        c.execute(f"INSERT INTO {name} (id, details) VALUES (?, ?)", (neuron_id, json.dumps(details)))

print("Writing examples")

c.execute(f'''CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, details TEXT)''')

for i, example in enumerate(examples):
    c.execute(f"INSERT INTO examples (id, details) VALUES (?, ?)", (i, json.dumps(example)))

print("Writing activations")

c.execute(f'''CREATE TABLE IF NOT EXISTS activations (id TEXT PRIMARY KEY, details BLOB)''')
for i, tensor in enumerate(activations):
    c.execute(f"INSERT INTO activations (id, details) VALUES (?, ?)", (i, sqlite3.Binary(tensor.numpy().tobytes())))

print("Writing importances")

c.execute(f'''CREATE TABLE IF NOT EXISTS importances (id TEXT PRIMARY KEY, details BLOB)''')
for i, tensor in enumerate(importances):
    c.execute(f"INSERT INTO importances (id, details) VALUES (?, ?)", (i, sqlite3.Binary(tensor.to(torch.float16).numpy().tobytes())))

conn.commit()
conn.close()


import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler

# For a full training on the msmarco dataset we decided to use the
# msmarco-passage_train_triples-small dataset.
# This is already provided by the huggingface datasets library.
# https://huggingface.co/datasets/irds/msmarco-passage_train_triples-small


# https://huggingface.co/docs/datasets/tabular_load#pandas-dataframes
# load Dataset from Pandas DataFrame
raw_input = pd.read_json("../data/triples-ms-marco-tiny.jsonl.gz", encoding="utf-8", lines=True)
df = []

for _, i in raw_input.iterrows():
    df += [
        {'query': i["query"],
         'text': i["positive_document"],
         'label': 1.0},
        {'query': i["query"],
         'text': i["negative_passage"],
         'label': 0.0},
    ]

training_data = Dataset.from_pandas(pd.DataFrame(df))

# The model we want to fine-tune
checkpoint = "prajjwal1/bert-tiny"
train_batch_size = 200
num_epochs = 10

# We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = CrossEncoder(checkpoint, num_labels=1, device=device)

train_dataloader = DataLoader(training_data, shuffle=True, batch_size=train_batch_size)

# We define our loss function
loss_function = torch.nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.model.parameters(), lr=2e-5)


num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.model.train()

for epoch in range(num_epochs):
    avg_train_loss = 0
    for batch in train_dataloader:

        # TODO: data should be formatted at the beginning of this script
        # problem: DataLoader doesn't like arrays
        formatted = []
        for i in range(len(batch["query"])):
            formatted += [
                [batch["query"][i], batch["text"][i]]
            ]
        outputs = model.predict(formatted, convert_to_tensor=True)
        inputs = torch.tensor(outputs, requires_grad=True)  # needed for backpropagation
        batch["label"] = torch.tensor(batch["label"]).to(device)  # needed for loss function

        loss = loss_function(inputs, batch["label"])
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        avg_train_loss += loss
    print(f"Average training loss: {avg_train_loss:.4f}")


# Save latest model
model.save("cross-encoder-custom-bert-tiny-512")

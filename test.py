import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. Data Preparation
# Let's create a small dummy dataset for lemmatization

data = [
    ("running", "run"),
    ("jumped", "jump"),
    ("flies", "fly"),
    ("walked", "walk"),
    ("singing", "sing"),
    ("played", "play"),
    ("eating", "eat"),
    ("speaking", "speak"),
    ("writing", "write"),
    ("dancing", "dance")
]

df = pd.DataFrame(data, columns=['inflected', 'lemma'])

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Preprocessing

# Create character to index mappings
chars = set(''.join(df['inflected'].tolist() + df['lemma'].tolist()))
char2idx = {char: idx for idx, char in enumerate(sorted(chars), start=1)}
char2idx['<PAD>'] = 0
idx2char = {idx: char for char, idx in char2idx.items()}


# Convert words to indices
def word_to_indices(word, max_len=20):
    return [char2idx[char] for char in word] + [0] * (max_len - len(word))


# 3. Dataset and DataLoader

class LemmaDataset(Dataset):
    def __init__(self, dataframe):
        self.inflected = [word_to_indices(word) for word in dataframe['inflected']]
        self.lemma = [word_to_indices(word) for word in dataframe['lemma']]

    def __len__(self):
        return len(self.inflected)

    def __getitem__(self, idx):
        return torch.tensor(self.inflected[idx]), torch.tensor(self.lemma[idx])


train_dataset = LemmaDataset(train_df)
val_dataset = LemmaDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)


# 4. Simple Seq2Seq Model (we'll replace this with Transformer variants later)

class Seq2SeqLemmatizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, trg):
        batch_size = src.shape[0]

        # Encoder
        embedded = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded)

        # Reshape hidden and cell states
        hidden = hidden.view(1, batch_size, self.hidden_size)
        cell = cell.view(1, batch_size, self.hidden_size)

        # Decoder
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, len(char2idx)).to(src.device)

        decoder_input = self.embedding(trg[:, 0]).unsqueeze(1)

        for t in range(trg_len):
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t] = prediction

            # Teacher forcing
            decoder_input = self.embedding(trg[:, t]).unsqueeze(1)

        return outputs


# 5. Training Loop

model = Seq2SeqLemmatizer(len(char2idx), 64, len(char2idx))
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for src, trg in train_loader:
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output.view(-1, len(char2idx)), trg.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, trg in val_loader:
            output = model(src, trg)
            loss = criterion(output.view(-1, len(char2idx)), trg.view(-1))
            val_loss += loss.item()

    print(
        f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')


# 6. Inference

def lemmatize(word):
    model.eval()
    indices = torch.tensor([word_to_indices(word)]).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(indices, indices)  # Using same input for target (we'll ignore it)
    predicted_indices = output.argmax(2).squeeze()
    return ''.join([idx2char[idx.item()] for idx in predicted_indices if idx.item() != 0])

# Test the model
test_words = ["walking", "speaks", "danced"]
for word in test_words:
    print(f"{word} -> {lemmatize(word)}")

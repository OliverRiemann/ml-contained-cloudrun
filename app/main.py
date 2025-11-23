from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re

app = FastAPI()

def tokenize(code: str):
    return re.findall(r'\w+|[^\s\w]', code)

def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def prepare_input(code_str: str, vocab: dict, max_len: int = 500):
    tokens = tokenize(code_str)
    token_ids = tokens_to_ids(tokens[:max_len], vocab)
    seq = torch.tensor(token_ids, dtype=torch.long)

    if len(seq) < max_len:
        pad_len = max_len - len(seq)
        pad_tensor = torch.full((pad_len,), vocab["<PAD>"], dtype=torch.long)
        seq = torch.cat([seq, pad_tensor])

    return seq.unsqueeze(0)

class CodeClassifierBiLSTM_Attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        return self.fc(context)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("app/vocab.pickle", "rb") as f:
    vocab = pickle.load(f)

with open("app/label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

model = CodeClassifierBiLSTM_Attn(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=256,
    num_classes=len(label_encoder.classes_),
    dropout=0.3
).to(device)

model.load_state_dict(torch.load("app/best_model.pt", map_location=device))
model.eval()

class CodeInput(BaseModel):
    code: str

@app.post("/predict")
def predict_language(data: CodeInput):
    try:
        input_tensor = prepare_input(data.code, vocab).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = output.argmax(dim=1).item()
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

        return {"predicted_language": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
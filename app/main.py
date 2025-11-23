from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder
import re

app = FastAPI()

label_encoder = LabelEncoder()

def tokenize(code):
    return re.findall(r'\w+|[^\s\w]', code)

def tokens_to_ids(tokens, vocab):
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

class CodeClassifierBiLSTM_Attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)  # Attention layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)                           
        attn_weights = torch.softmax(self.attn(out), dim=1) 
        context = torch.sum(attn_weights * out, dim=1)  
        context = self.dropout(context)
        return self.fc(context)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)

def prepare_input(code_str, vocab, max_len=500):
    tokens = tokenize(code_str)
    token_ids = tokens_to_ids(tokens[:max_len], vocab)
    seq = torch.tensor(token_ids, dtype=torch.long)

    if len(seq) < max_len:
        pad_len = max_len - len(seq)
        pad_tensor = torch.full((pad_len,), vocab["<PAD>"], dtype=torch.long)
        seq = torch.cat([seq, pad_tensor])
    return seq.unsqueeze(0)

model = CodeClassifierBiLSTM_Attn(
	vocab_size=len(vocab),
	embed_dim=128,
	hidden_dim=256,
	num_classes=len(label_encoder.classes_),
	dropout=0.3
).to(device)

model = torch.load("app/best_model.pt", map_location=torch.device('cpu'))
model.eval()

# Input model schema
class CodeInput(BaseModel):
    code: str

@app.post("/predict")

def predict_language(data: CodeInput):
    try:
        input_code = data.code
        # Replace this with your actual preprocessing & model logic
        # Example: preprocess input_code and predict
        with torch.no_grad():
            output = model(input_code)  # change based on your model
            predicted = output.argmax().item()

        return {"language_id": predicted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
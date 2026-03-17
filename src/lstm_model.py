import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embed = self.embedding(x)          # (batch, seq_len, embed_dim)
        out, _ = self.lstm(embed)          # (batch, seq_len, hidden_dim)
        logits = self.fc(out)              # (batch, seq_len, vocab_size)
        return logits
    
    def generate(self, input_ids, idx_to_word, max_new_tokens=5, eos_idx=1):
        self.eval()
        with torch.no_grad():
            ids = input_ids.clone()
            for _ in range(max_new_tokens):
                logits = self.forward(ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_token], dim=1)
                if next_token.item() == eos_idx:
                    break
            
            # Переводим в слова
            words = [idx_to_word.get(i.item(), '') for i in ids[0]]
            return ' '.join(words)

if __name__ == '__main__':
    # Быстрая проверка что модель работает
    vocab_size = 44560
    model = LSTMModel(vocab_size)
    
    x = torch.randint(0, vocab_size, (2, 10))
    out = model.forward(x)
    print(f"Forward output shape: {out.shape}")
    
    idx_to_word = {i: f"word{i}" for i in range(vocab_size)}
    result = model.generate(x[:1], idx_to_word)
    print(f"Generated: {result}")
    print("Модель работает!")
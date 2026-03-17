import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class NextTokenDataset(Dataset):
    def __init__(self, csv_path, vocab=None, max_len=20, max_rows=None):
        df = pd.read_csv(csv_path, nrows=max_rows)
        self.max_len = max_len
        
        # Собираем все токены
        all_tokens = []
        for text in df['text'].dropna():
            all_tokens.extend(text.split())
        
        # Создаём словарь если не передан
        if vocab is None:
            unique_words = list(set(all_tokens))
            self.vocab = {'<pad>': 0, '<eos>': 1}
            for word in unique_words:
                self.vocab[word] = len(self.vocab)
        else:
            self.vocab = vocab
            
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Создаём примеры X -> Y
        self.samples = []
        for text in df['text'].dropna():
            tokens = text.split()
            if len(tokens) < 2:
                continue
            ids = [self.vocab.get(t, 0) for t in tokens[:max_len]]
            self.samples.append(ids)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys

def get_dataloaders(batch_size=256):
    train_dataset = NextTokenDataset('data/train.csv', max_rows=50000)
    val_dataset = NextTokenDataset('data/val.csv', vocab=train_dataset.vocab)
    test_dataset = NextTokenDataset('data/test.csv', vocab=train_dataset.vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            collate_fn=collate_fn)
    
    print(f"Vocab size: {len(train_dataset.vocab)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader, train_dataset.vocab

if __name__ == '__main__':
    train_loader, val_loader, test_loader, vocab = get_dataloaders()
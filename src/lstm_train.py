import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import LSTMModel
from next_token_dataset import NextTokenDataset, collate_fn
from rouge_score import rouge_scorer

def evaluate_rouge(model, dataset, idx_to_word, num_samples=100):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    scores1, scores2 = [], []
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            ids = dataset.samples[i]
            if len(ids) < 4:
                continue
            
            # Берём первые 3/4 как вход
            split = max(1, len(ids) * 3 // 4)
            input_ids = torch.tensor(ids[:split]).unsqueeze(0)
            target_words = ' '.join([idx_to_word.get(id, '') 
                                    for id in ids[split:]])
            
            generated = model.generate(input_ids, idx_to_word, 
                                      max_new_tokens=len(ids)-split)
            # Берём только новые слова
            input_words = ' '.join([idx_to_word.get(id, '') 
                                   for id in ids[:split]])
            new_words = generated[len(input_words):].strip()
            
            score = scorer.score(target_words, new_words)
            scores1.append(score['rouge1'].fmeasure)
            scores2.append(score['rouge2'].fmeasure)
    
    return sum(scores1)/len(scores1) if scores1 else 0, \
           sum(scores2)/len(scores2) if scores2 else 0

def train(num_epochs=5, batch_size=256, lr=0.001):
    # Загружаем данные
    print("Загружаем данные...")
    train_dataset = NextTokenDataset('data/train.csv', max_rows=50000)
    val_dataset = NextTokenDataset('data/val.csv', 
                                   vocab=train_dataset.vocab, max_rows=5000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(train_dataset.vocab)
    idx_to_word = {v: k for k, v in train_dataset.vocab.items()}
    
    # Создаём модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    model = LSTMModel(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Обучение
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        r1, r2 = evaluate_rouge(model, val_dataset, idx_to_word)
        print(f"\n--- Epoch {epoch+1} ---")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}\n")
        
        # Пример предсказания
        sample_ids = val_dataset.samples[0]
        split = len(sample_ids) * 3 // 4
        input_ids = torch.tensor(sample_ids[:split]).unsqueeze(0).to(device)
        generated = model.generate(input_ids, idx_to_word)
        print(f"Пример: {generated}\n")
    
    # Сохраняем модель
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_model.pt')
    print("Модель сохранена в models/lstm_model.pt")
    
    return model, train_dataset.vocab

if __name__ == '__main__':
    train()
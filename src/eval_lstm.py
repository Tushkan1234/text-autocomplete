import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import LSTMModel
from next_token_dataset import NextTokenDataset
from rouge_score import rouge_scorer

def evaluate(model_path='models/lstm_model.pt', num_samples=200):
    # Загружаем данные
    print("Загружаем данные...")
    train_dataset = NextTokenDataset('data/train.csv', max_rows=50000)
    test_dataset = NextTokenDataset('data/test.csv', 
                                    vocab=train_dataset.vocab, max_rows=5000)
    
    idx_to_word = {v: k for k, v in train_dataset.vocab.items()}
    vocab_size = len(train_dataset.vocab)
    
    # Загружаем модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Модель загружена!")
    
    # Считаем ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    scores1, scores2 = [], []
    
    print(f"\nПримеры предсказаний:")
    print("-" * 50)
    
    for i in range(min(num_samples, len(test_dataset))):
        ids = test_dataset.samples[i]
        if len(ids) < 4:
            continue
        
        split = max(1, len(ids) * 3 // 4)
        input_ids = torch.tensor(ids[:split]).unsqueeze(0).to(device)
        
        input_text = ' '.join([idx_to_word.get(id, '') for id in ids[:split]])
        target_text = ' '.join([idx_to_word.get(id, '') for id in ids[split:]])
        
        generated = model.generate(input_ids, idx_to_word, 
                                   max_new_tokens=len(ids)-split)
        new_words = generated[len(input_text):].strip()
        
        score = scorer.score(target_text, new_words)
        scores1.append(score['rouge1'].fmeasure)
        scores2.append(score['rouge2'].fmeasure)
        
        # Выводим первые 5 примеров
        if i < 5:
            print(f"Вход:      {input_text}")
            print(f"Цель:      {target_text}")
            print(f"Предсказание: {new_words}")
            print(f"ROUGE-1: {score['rouge1'].fmeasure:.4f}")
            print("-" * 50)
    
    r1 = sum(scores1) / len(scores1) if scores1 else 0
    r2 = sum(scores2) / len(scores2) if scores2 else 0
    print(f"\nИтоговые метрики на тесте:")
    print(f"ROUGE-1: {r1:.4f}")
    print(f"ROUGE-2: {r2:.4f}")
    
    return r1, r2

if __name__ == '__main__':
    evaluate()
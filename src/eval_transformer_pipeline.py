import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import pipeline
from next_token_dataset import NextTokenDataset
from rouge_score import rouge_scorer

def evaluate_transformer(num_samples=100):
    print("Загружаем модель distilgpt2...")
    generator = pipeline("text-generation", model="distilgpt2")
    
    print("Загружаем данные...")
    train_dataset = NextTokenDataset('data/train.csv', max_rows=50000)
    test_dataset = NextTokenDataset('data/test.csv',
                                    vocab=train_dataset.vocab, max_rows=1000)
    
    idx_to_word = {v: k for k, v in train_dataset.vocab.items()}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    scores1, scores2 = [], []
    
    print("\nПримеры предсказаний:")
    print("-" * 50)
    
    for i in range(min(num_samples, len(test_dataset))):
        ids = test_dataset.samples[i]
        if len(ids) < 4:
            continue
        
        split = max(1, len(ids) * 3 // 4)
        input_text = ' '.join([idx_to_word.get(id, '') for id in ids[:split]])
        target_text = ' '.join([idx_to_word.get(id, '') for id in ids[split:]])
        
        # Генерируем продолжение
        result = generator(input_text, 
                          max_length=len(ids) + 5,
                          do_sample=False,
                          pad_token_id=50256)
        
        generated_full = result[0]['generated_text']
        new_words = generated_full[len(input_text):].strip()
        
        score = scorer.score(target_text, new_words)
        scores1.append(score['rouge1'].fmeasure)
        scores2.append(score['rouge2'].fmeasure)
        
        # Выводим первые 5 примеров
        if i < 5:
            print(f"Вход:         {input_text}")
            print(f"Цель:         {target_text}")
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
    evaluate_transformer()
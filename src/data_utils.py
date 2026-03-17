import kagglehub
import pandas as pd
import re
import os

def download_dataset():
    path = kagglehub.dataset_download("kazanova/sentiment140")
    # Находим csv файл
    for f in os.listdir(path):
        if f.endswith('.csv'):
            return os.path.join(path, f)

def clean_text(text):
    text = text.lower()  # нижний регистр
    text = re.sub(r'http\S+', '', text)  # удаляем ссылки
    text = re.sub(r'@\w+', '', text)  # удаляем упоминания
    text = re.sub(r'[^\w\s]', '', text)  # удаляем спецсимволы
    text = re.sub(r'\s+', ' ', text).strip()  # лишние пробелы
    return text

def load_and_clean_data(save_path='data/dataset_processed.csv'):
    print("Скачиваем датасет...")
    csv_path = download_dataset()
    
    print("Загружаем данные...")
    df = pd.read_csv(csv_path, encoding='latin-1', header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    
    print("Очищаем тексты...")
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]  # убираем пустые строки
    
    os.makedirs('data', exist_ok=True)
    df[['text']].to_csv(save_path, index=False)
    print(f"Сохранено в {save_path}, строк: {len(df)}")
    return df

if __name__ == '__main__':
    df = load_and_clean_data()
    print(df['text'].head(10))


from sklearn.model_selection import train_test_split

def tokenize_and_split(df, save_dir='data/'):
    print("Токенизируем...")
    df['tokens'] = df['text'].apply(lambda x: x.split())
    df = df[df['tokens'].apply(len) > 3]  # убираем слишком короткие
    
    # Разбивка 80/10/10
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train[['text']].to_csv(save_dir + 'train.csv', index=False)
    val[['text']].to_csv(save_dir + 'val.csv', index=False)
    test[['text']].to_csv(save_dir + 'test.csv', index=False)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

if __name__ == '__main__':
    df = load_and_clean_data()
    train, val, test = tokenize_and_split(df)
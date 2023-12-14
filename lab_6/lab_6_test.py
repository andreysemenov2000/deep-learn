import torch
from transformers import BertTokenizer, BertForSequenceClassification


def test_model():
    # Проверим работу модели
    tokenizer = BertTokenizer.from_pretrained('comment_classifier')
    model = BertForSequenceClassification.from_pretrained('comment_classifier')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Пример текста для проверки
    sample_text = "Может кому-то и не понравилось, мне кажется очень неплохо. Ставлю лайк"
    # Токенизация и подготовка текста для модели
    encoded_text = tokenizer(sample_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    # Получение предсказания
    logits = output.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Вывод результата
    if predicted_class == 1:
        print("Текст токсичен.")
    else:
        print("Текст не токсичен.")


test_model()
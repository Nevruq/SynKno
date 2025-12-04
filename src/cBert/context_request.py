import wikipedia
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
import cBert.wrapper_CBert as cBert

def predict_context_needed(text, model, tokenizer, device, max_length=128):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    model = cBert.cBERT('google-bert/bert-base-multilingual-cased', 2).to(device)
    model.load_state_dict(torch.load('src/data/model/cBERT_35k.pth', device))

    test_text = "I need information about red indian birds!"
    context_needed = predict_context_needed(test_text, model, tokenizer, device)
    print(f"Predicted context: {context_needed}")
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    print(preds.item())
    return "context_needed" if preds.item() == 1 else "context_not_needed"
# Predicted context: context_needed 



if __name__ == "__main__":
    pass
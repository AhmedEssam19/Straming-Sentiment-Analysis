import torch

from sentiment_analyzer import SentimentAnalyzer


class ModelWrapper:
    def __init__(self):
        model_name = 'bert-base-uncased'
        self._tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model = SentimentAnalyzer(model_name=model_name, output_units=1, dropout=0.1)
        self._model.load_state_dict(torch.load('model.pth', map_location=self._device))
        self._model.to(self._device)
        self._model.eval()

    def predict(self, text):
        with torch.no_grad():
            encoded_text = self._tokenizer(text, padding='max_length', max_length=64, add_special_tokens=True,
                                           truncation='longest_first', return_tensors='pt')
            input_ids = encoded_text['input_ids'].to(self._device)
            attention_mask = encoded_text['attention_mask'].to(self._device)
            logits = self._model(input_ids, attention_mask)
            prob = torch.sigmoid(logits).cpu().item()

            if 0.4 <= prob <= 0.6:
                return 'Neutral'
            elif prob > 0.6:
                return 'Positive'
            else:
                return 'Negative'


def test():
    import pandas as pd
    model = ModelWrapper()
    df_val = pd.read_csv('test.csv')
    df_val['pred'] = df_val['text'].apply(model.predict)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(df_val['target'], df_val['pred']))


# test()

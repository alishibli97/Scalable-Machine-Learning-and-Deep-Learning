from torch import nn

from transformers import BertTokenizer, BertModel

class SBERT(nn.Module):
    def __init__(self):
        super(SBERT, self).__init__()

        # self.bert = ## NEED TO INITIALIZE BERT HERE

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        return output


if __name__=="__main__":
    text = "Replace me by any text you'd like."

    sbert = SBERT()

    # sbert.forward(text)
    print(sbert(text).shape)
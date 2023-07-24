from transformers import AutoTokenizer, AlbertForMaskedLM, BertTokenizer, BertForMaskedLM
import torch
# from torch.nn.functional import softmax
# from transformers import AutoTokenizer, TFAutoModel

# pretrained = 'voidful/albert_chinese_base'
# tokenizer = AutoTokenizer.from_pretrained(pretrained)
# model = AlbertForMaskedLM.from_pretrained(pretrained)

# model = TFAutoModel.from_pretrained("./chinese-roberta-wwm-ext")
# tokenizer = AutoTokenizer.from_pretrained("./chinese-roberta-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("./chinese-roberta-wwm-ext")
model = BertForMaskedLM.from_pretrained("./chinese-roberta-wwm-ext")

def predict(inputtext):
    maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

    input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=input_ids)
    loss, prediction_scores = outputs[:2]
    # logit_prob = softmax(prediction_scores[0, maskpos], dim=-1).data.tolist()
    # predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_index = torch.topk(prediction_scores[0, maskpos], k=20).indices.tolist()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    predicted_token = [tokenizer.convert_ids_to_tokens([idx])[0] for idx in predicted_index]

    # print(predicted_token, logit_prob[predicted_index])
    return predicted_token


if __name__ == '__main__':
    inputtext = "这辆汽车很[MASK]油。"
    for i in range(inputtext.count('[MASK]')):
        predicted = predict(inputtext)
        # inputtext = inputtext.replace("[MASK]", predicted, 1)
        inputtext = [inputtext.replace("[MASK]", word, 1) for word in predicted]
        # print(predicted, '==>', inputtext)

    print('final result:',  *inputtext, sep='\n') # * symbol is used to parse out list elements.

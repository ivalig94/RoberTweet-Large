# RoberTweet-Large

RoberTweet-Large es un modelo capaz de clasificar tweets en Inglés a 3 niveles: positive, negative y neutral. Para realizar predicciones con el, basta con realizar los siguientes pasos:


-->Para utilizar este modelo, basta con copiar el fragmento de codigo a continuación modificando los siguientes parametros y teniendo en cuenta lo siguiente:
-->El archivo que se le pase al modelo, debe ser un archivo ".csv" que contenga los tweets en una cabecera llamada "review".
-->En "file_path"--> Indicaremos la ruta donde hemos guardado el archivo .pt que nos hemos descargado desde este repositorio y que contiene el modelo.
-->En "Tweets"--> Indicaremos la ruta donde se encuentra el archivo .csv que contiene los tweets que se quieren clasificar.
-->Los resultados se encontraran en un array llamado "preds" que contendra la etiqueta asignada a cada tweet en el mismo orden que se encuentran los tweets en el archivo "Tweets". 
-->Este array se puede postprocesar de la manera que se desee para trabajar con los resultados.




!pip install transformers
from transformers.utils.dummy_pt_objects import PreTrainedModel
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report

from google.colab import drive
drive.mount('/content/drive')


###################  RELLENAR LOS VALORES DE ESTAS 2 VARIABLES CON LOS DATOS DESCRITOS ARRIBA  ##########################

file_path= ''
tweets= pd.read_csv("",encoding = 'utf8')

#########################################################################################################################



def create_dataloader_seq(inputs,masks,batch_size):
  # Create the DataLoader for our data set
  data =  TensorDataset(inputs, masks)
  sampler=SequentialSampler(data)
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader


def load_checkpoint(load_path, model):
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def preprocessdata_tensortokenids_masks_usingencode_plus(data,tokenizer,MAX_LEN):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

class ROBERTAClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.1):
        super(ROBERTAClassifier, self).__init__()
                # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        H=50
        D_out=n_classes
        D_in=1024
        self.roberta = RobertaModel.from_pretrained('roberta-large',return_dict=False)
        self.classifier = nn.Sequential(
        nn.Linear(D_in, D_in),
        nn.Dropout(dropout_rate),
        nn.Linear(D_in, D_out)
        )

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.roberta(input_ids=input_ids,attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []
    predictions = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    print('all_logits:')
    print(all_logits)
    if NCLASES == 2:
      # Apply softmax to calculate probabilities:
      probs = F.softmax(all_logits, dim=1).cpu().numpy()
      preds = np.where(probs[:, 1] > THRESHOLD, 1, 0)
    else:
      _,predicts=torch.max(all_logits, dim=1)
      print('predicts:')
      print(predicts)
      predictions.extend(predicts)
      preds = torch.stack(predictions).cpu()
      #probs = F.softmax(all_logits, dim=1)
      #preds = torch.argmax(all_logits, dim=1).flatten()
    return preds


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NCLASES=3
device = torch.device("cpu")
X_test=tweets['review']


model=ROBERTAClassifier()
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
MAX_LEN=128
test_token_ids_tensors,test_masks_tensors=preprocessdata_tensortokenids_masks_usingencode_plus(X_test,tokenizer,MAX_LEN)
test_dataloader_seq=create_dataloader_seq(test_token_ids_tensors,test_masks_tensors,1)
load_checkpoint(file_path + '/model.pt', model)
preds = bert_predict(model, test_dataloader_seq)

  
prediccionesTensor= preds.numpy()[0:]
predicciones_letras=[]

for value in prediccionesTensor:
  if value==2:
    predicciones_letras.append('positive')
  elif value==1:
    predicciones_letras.append('neutral')
  elif value==0:
    predicciones_letras.append('negative')   

#print(prediccionesTensor)
print(predicciones_letras)

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torchcrf import CRF
from data_process import role2label, trigger2label,label2role,label2trigger
from data_process import train_loader, dev_loader, test_loader
import json

NUM_TAGS = len(role2label)
NUM_RELATIONS = len(trigger2label)

root_path = 'chinese_roberta/'
from transformers import BertModel


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(torch.nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


def viterbi_decode(emissions,transitions,length,num_tags=NUM_TAGS,k=5):
    prev = torch.zeros(length,num_tags)
    probs = torch.zeros(length,num_tags)
    for i in range(length):
        if i == 0:
            for j in range(num_tags):
                prev[0][j] = -1
                probs[0][j] = torch.log(emissions[0][j])
        else:
            for j in range(num_tags):
                res = []
                for k in range(num_tags):
                    probs[i][j] = torch.log(probs[i-1][k]) + torch.log(transitions[k][j]) + torch.log(emissions[i][j])
                    res.append((k,probs[i][j]))
                sorted(res,key=lambda x:x[1],reverse=True)
                prev[i][j] = res[0][0]

    seq_res = []
    for j in range(NUM_TAGS):
        last = length - 1
        score = -1e10
        res = []
        while last!=-1:
            if last == length-1:
                score = probs[last][j]
                prev = j
            res.append(trigger2label[prev])
            prev = prev[last][prev]
            last -= 1
        seq_res.append((score,res))
    sorted(seq_res,key=lambda x:x[0])
    if len(seq_res) < k:
        return seq_res
    else:
        return seq_res[:k]







class JointModel(pl.LightningModule):
    def __init__(self, root_path, num_layers, hidden_size, num_tags, num_relations, bidirectional=True, dropout=0.3,
                 soft_embedding_dim=1024):
        super().__init__()
        self.model = BertModel.from_pretrained(root_path)
        self.lstm = torch.nn.LSTM(1024, hidden_size, batch_first=True,  bidirectional=bidirectional)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.num_tags = num_tags
        self.num_relations = num_relations
        self.embedding = torch.nn.Embedding(num_tags, embedding_dim=soft_embedding_dim)

        ## project_layer
        self.dense = torch.nn.Linear(self.num_directions * self.hidden_size, self.num_tags)
        ### crf decode layer
        self.crf = CRF(self.num_tags, batch_first=True)

        self.subject_dense = torch.nn.Linear(soft_embedding_dim, num_relations)
        self.object_dense = torch.nn.Linear(soft_embedding_dim, num_relations)

        self.relation_dense = torch.nn.Linear(num_relations*num_relations,num_relations)

        self.doc_dense = torch.nn.Sequential(torch.nn.Linear(1024, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10),
                                             torch.nn.Softmax(dim=-1))



    def training_step(self, batch, batch_idx):
        input, label, relation_matrix, classes, length = batch
        maxlen = input.shape[-1]
        emissions, predicted_relation_matrix, cls_ = self(input)
        tags = label.long()
        mask = self.sequence_mask(length,max_len=maxlen)
        log_loss = -self.crf(emissions, tags, mask=mask)
        rel_loss = MulticlassDiceLoss()(predicted_relation_matrix,relation_matrix)
        classes = classes.squeeze(dim=1)
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_,classes.float())

        return {'loss': log_loss + rel_loss + cls_loss}

    def validation_step(self,batch,batch_idx):
        output = self.training_step(batch,batch_idx)
        val_loss = output['loss']
        return {'val_loss':val_loss}

    def validation_epoch_end(self,outputs):
        outputs = torch.stack([x['val_loss'] for x in outputs]).mean()
        print("validation_loss: {}".format(outputs.data))
        return {'val_loss':outputs.data}

    def test_step(self, batch, batch_idx):
        input, id, length,original_text = batch
        maxlen = input.shape[-1]
        mask = self.sequence_mask(length,max_len=maxlen)
        emissions, predicted_relation_matrix, cls = self(input)

        return {'emissions':emissions, 'predicted_relation_matrix':predicted_relation_matrix, 'cls':cls,'length':length,'id':id,'original_text':original_text}

    def write_by_line(self,path, data, t_code="utf-8"):
        with open(path,'w') as outfile:
            outfile.write(data.encode(t_code)+'\n')


    def test_step_end(self, outputs):
        emissions = outputs['emissions']
        predicted_relation_matrix = outputs['predicted_relation_matrix']
        cls = outputs['cls']
        length = outputs['length']
        id = outputs['id']
        original_text = outputs['original_text']
        if emissions.shape == 4:
            emissions = emissions.squeeze(dim=0)
        decoded_sequence = viterbi_decode(emissions,self.crf.transitions,length=length,num_tags=NUM_TAGS)
        predicted_relation_matrix = torch.round(predicted_relation_matrix)
        events = {}
        events['text'] = original_text
        events['id'] = id
        events['event_list'] = []
        for i in range(length):
            for j in range(length):
                for k in range(NUM_RELATIONS):
                    if predicted_relation_matrix[i][j][k]>0.8:
                        event = {}
                        event_type = label2trigger[k]
                        event['event_type'] = event_type
                        event['arguments'] = []
                        for seq in decoded_sequence:
                            start_i = i
                            start_j = j
                            while start_i>=0:
                                if seq[start_i]=='O':
                                    continue
                                elif seq[start_i].start_with('B-'):
                                    role = seq[start_i][2:]
                                    argument = original_text[start_i:i+1]
                                    event['arguments'].append({'role':role,'argument':argument})
                            while start_j >=0:
                                if seq[start_j] == 'O':
                                    continue
                                elif seq[start_j].start_with('B-'):
                                    role = seq[start_j][2:]
                                    argument = original_text[start_j:j+1]
                                    event['arguments'].append({'role':role,'argument':argument})
                            events['event_list'].append(event)
        data = json.dumps(events)
        self.write_by_line('test1_pred.json',data)
        return {}

    def sequence_mask(self,sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.range(0, max_len - 1).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        # seq_range_expand = torch.tensor(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        max_len = inputs.shape[-1]
        inputs = inputs.squeeze(dim=1)
        words_embedding, seq_embedding = self.model(inputs)
        ## words_embedding: batch_size, seq_len,1024
        output, (hn, cn) = self.lstm(words_embedding)
        ## output: batch_size seq_len, num_directions * hidden_size
        ## hn: batch, num_layers*num_directions, hidden_size
        ## cn: batch, num_layers*num_directions, hidden_size
        ## num_tag: 243, num_relations: 66
        emissions = self.dense(output)

        emissions = torch.nn.functional.softmax(emissions, dim=-1)
        soft_label = torch.einsum('bsn,nf->bsnf', emissions, self.embedding.weight)
        ### batch seqlen embedding_dim
        soft_label = soft_label.mean(dim=2)
        ### batch seqlen relation_num
        subject_embedding = self.subject_dense(soft_label)
        ### batch seqlen relation_num
        object_embedding = self.object_dense(soft_label)
        predicted_relation_matrix = torch.einsum('bsr,bfh->bsfrh', subject_embedding, object_embedding)
        predicted_relation_matrix = predicted_relation_matrix.reshape(batch_size,max_len,max_len,-1)
        predicted_relation_matrix = self.relation_dense(predicted_relation_matrix)

        cls_ = self.doc_dense(seq_embedding)

        return emissions, predicted_relation_matrix, cls_

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return dev_loader

    def test_loader(self):
        return test_loader

model = JointModel(root_path=root_path,num_layers=2, hidden_size=1024, num_tags=NUM_TAGS, num_relations=NUM_RELATIONS)
trainer = Trainer()
trainer.fit(model)


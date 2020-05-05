#coding:utf8
import json
import numpy as np
import torch

def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w") as outfile:
        [outfile.write(d.encode(t_code) + "\n") for d in data]


def data_process(path,is_predict=False):
    """data_process"""

    max_len = 0

    with open(path) as f:
        data = []
        for line in f:
            d_json = json.loads(line.strip())
            _id = d_json["id"]

            text_a = [
                u"，" if t == u" " or t == u"\n" or t == u"\t" else t
                for t in list(d_json["text"].lower())
            ]
            if is_predict:
                data.append((text_a,d_json['id'],d_json['text']))
            else:
                max_len = max(max_len, len(text_a))
                # print("len text",len(d_json['text'].lower()))
                # print("len text_a",len(text_a))

                labels = ['O'] * len(text_a)
                relations = [['N'] for i in range(len(labels))]
                matched_id = [[i] for i in range(len(labels))]
                # print("len label: ",len(labels))
                # print("len relations",len(relations))
                # print("len match_id",len(matched_id))
                classes = []

                for event in d_json['event_list']:
                    d_type = event['event_type']
                    cls = event['class']
                    if cls not in classes:
                        classes.append(cls)
                    for i in range(len(event['arguments'])):

                        argument = event['arguments'][i]
                        argument_start_index = argument['argument_start_index']
                        argument_role = argument['role']
                        argument_word = argument['argument']
                        labels[argument_start_index:argument_start_index+len(argument_word)] = ['I-{}'.format(argument_role) for i in range(len(argument_word))]
                        # print(text_a)
                        # print("labels: ",len(labels),"argument_start_index: ",argument_start_index)
                        labels[argument_start_index] = 'B-{}'.format(argument_role)
                        if i+1<len(event['arguments']):
                            matched_argument = event['arguments'][i+1]
                            matched_argument_start_index = matched_argument['argument_start_index']
                            matched_argument_word = matched_argument['argument']
                            if len(matched_id[argument_start_index+len(argument_word)-1])==1 and matched_id[argument_start_index+len(argument_word)-1][0]==argument_start_index+len(argument_word)-1:
                                matched_id[argument_start_index + len(argument_word)-1] = []
                                relations[argument_start_index + len(argument_word)-1] = []
                            matched_id[argument_start_index + len(argument_word)-1].append(matched_argument_start_index + len(matched_argument_word)-1)
                            relations[argument_start_index + len(argument_word)-1].append(d_type)
                classes = classes if len(classes) else ['O']
                data.append((text_a,labels,relations,matched_id,classes))

    return data


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if u"B-{}".format(_type) not in labels:
            labels.extend([u"B-{}".format(_type), u"I-{}".format(_type)])
        return labels

    labels = []
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip())
            if model == u"trigger":
                labels = label_add(labels, d_json["event_type"])
            elif model == u"role":
                for role in d_json["role_list"]:
                    labels = label_add(labels, role["role"])
            elif model == u'class':
                if d_json['class'] not in labels:
                    labels.append(d_json['class'])
    labels.append(u"O")
    return labels


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


def predict_data_process(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["text"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)


schema_trigger = schema_process('data/event_schema.json',model='trigger')
schema_role = schema_process('data/event_schema.json',model='role')
schema_cls = schema_process('data/event_schema.json',model='class')

print(schema_cls)

sorted(schema_trigger,key=lambda x:len(x))
print(schema_role)

schema_trigger = list(set(['N' if word=='O' else word[2:] for word in schema_trigger]))



# print([role.split('\t')[0] for role in data_role])
train_data = data_process('data/train.json')
dev_data = data_process('data/dev.json')
test_data = data_process('data/test1.json',is_predict=True)

print(train_data[0])


role2label = dict([(role,i) for i,role in enumerate(schema_role)])
label2role =  dict([(i,role) for i,role in enumerate(schema_role)])
trigger2label = dict([trigger,i] for i,trigger in enumerate(schema_trigger))
label2trigger = dict([i,trigger] for i,trigger in enumerate(schema_trigger))
cls2label = dict([cls,i] for i,cls in enumerate(schema_cls))
label2cls = dict([i,cls] for i,cls in enumerate(schema_cls))
print(cls2label)


from torch.utils.data import Dataset, DataLoader

root_path = 'chinese_roberta_wwm_large_ext_pytorch/'
from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained(root_path)
# model = BertModel.from_pretrained(root_path)

# input = "今天天气真好鱿鱼"
# inputs = tokenizer.encode(input,return_tensors='pt',pad_to_max_length=True,add_special_tokens=False,max_length=20)
# print(inputs)
# output = model(inputs)[0]
# print(output.shape)

from sklearn.preprocessing import MultiLabelBinarizer


class NerDataset(Dataset):
    def __init__(self,data,role2label,trigger2label,cls2label,pad_length=400,bert_path=root_path):
        self.data = data
        self.role2label = role2label
        self.trigger2label = trigger2label
        self.cls2label = cls2label
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # self.model = BertModel.from_pretrained(root_path)
        self.total_role = len(role2label)
        self.total_trigger = len(trigger2label)
        self.max_length = pad_length
        role2label_bin = MultiLabelBinarizer()
        role2label_bin.fit([[i for i in range(len(role2label))]])
        self.role2label_bin = role2label_bin

        relation2label_bin = MultiLabelBinarizer()
        relation2label_bin.fit([[i for i in range(len(trigger2label))]])
        self.relation2label_bin = relation2label_bin

        ids2label_bin = MultiLabelBinarizer()
        ids2label_bin.fit([[i for i in range(self.max_length)]])
        self.ids2label_bin = ids2label_bin

        cls2label_bin = MultiLabelBinarizer()
        cls2label_bin.fit([[i for i in range(len(cls2label))]])
        self.cls2label_bin = cls2label_bin

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_ = self.data[index]
        length = len(data_[0])
        text_a = ''.join(data_[0])
        inputs = self.tokenizer.encode(text_a,return_tensors='pt',add_special_tokens=False,max_length=self.max_length,pad_to_max_length=True)

        label = data_[1] + ['O']*(self.max_length - len(data_[1]))
        label = np.array([role2label[role] for role in label])
        # label = self.role2label_bin.transform(label)

        relations = data_[2]
        matched_ids = data_[3]
        # relations = [ [trigger2label[i] for i in rel] for rel in relations]
        # relations = self.relation2label_bin.transform(relations)
        out = trigger2label['N']
        relation_matrix = torch.zeros(self.max_length,self.max_length,self.total_trigger)
        relation_matrix.fill_(out)
        for i in range(len(relations)):
            if relations[i] == ['N']:
                continue
            maid = matched_ids[i]
            relation = relations[i]
            for j in range(len(relation)):
                relation_matrix[i][maid[j]][trigger2label[relation[j]]] = 1
                relation_matrix[maid[j]][i][trigger2label[relation[j]]] = 1
        # matched_ids = data_[3] + [[ids + i] for i in range(self.max_length - ids)]
        # matched_ids = self.ids2label_bin.transform(matched_ids)
        classes = data_[4]
        classes = [cls2label[cls] for cls in classes ]
        classes = self.cls2label_bin.transform([classes])

        return inputs, label ,relation_matrix, classes, length

class LabelDataset(Dataset):
    def __init__(self,data,bert_path=root_path,pad_length = 400):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_length = pad_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_ = self.data[index]
        length = len(data_[0])
        id = data_[1]
        text_a = ''.join(data_[0])
        original_text = data_[2]
        inputs = self.tokenizer.encode(text_a, return_tensors='pt', add_special_tokens=False,max_length=self.max_length, pad_to_max_length=True)
        return inputs,id,length,original_text

### dataset
train_dataset = NerDataset(train_data,role2label,trigger2label,cls2label)
dev_dataset = NerDataset(dev_data,role2label,trigger2label,cls2label)
test_dataset = LabelDataset(test_data)

### loader
train_loader = DataLoader(train_dataset,batch_size=32)
dev_loader = DataLoader(dev_dataset,batch_size=32)
test_loader = DataLoader(test_dataset,batch_size=32)

if __name__ == '__main__':
    for batch_idx, sample in enumerate(test_loader):
        print(sample[0].shape)
        print(sample[1])
        print(sample[2].shape)
        print(sample[3])
        break







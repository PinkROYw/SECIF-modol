#! -*- coding:utf-8 -*-
import math
import pandas as pd
from PIL import Image
from bert4torch.tokenizers import Tokenizer
from torchvision.models import *
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from capsule2 import *

maxlen = 128
batch_size = 16
config_path = '/data/cing/premodel/bert/config.json'
checkpoint_path = '/data/cing/premodel/bert/pytorch_model.bin'
dict_path = '/data/cing/premodel/bert/vocab.txt'

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 固定seed
# seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
label2id = {}


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        data = json.load(open(filename))
        for d in tqdm(data[:5]):
            try:
                path = f'/data/cing/muti-data/data/' + d['id'] + '.jpg'
                img = Image.open(path)
            except:
                continue
            img = img.resize((512, 512))
            if len(img.split()) == 3:
                img.load()
                D.append((d['context'], img, int(d['label']) + 1))
        return D


# 分词转换为tensor格式
def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels, batch_imgs = [], [], [], []
    for text, img, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        img_ids = np.array(img).tolist()
        batch_imgs.append(img_ids)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_imgs = torch.tensor(batch_imgs, dtype=torch.float, device=device)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids, batch_imgs], batch_labels.flatten()


# 加载数据集
train_dataloader = DataLoader(MyDataset('./train.json'), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
valid_dataloader = DataLoader(MyDataset('./test.json'), batch_size=batch_size, collate_fn=collate_fn)


class Gathered_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 1
        self.attention_head_size = 768
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

        self.dense = nn.Linear(768, 768)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)

        return output


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)

        # attention
        self.att = Gathered_Attention()

        # 胶囊网络
        self.caps_layer = Caps_Layer(768, 128, 768, 2)

        self.full_resnet = resnet101(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-2]),
        )

        self.hidden_trans = nn.Sequential(
            nn.Conv2d(self.full_resnet.fc.in_features, 64, 1),
            nn.Flatten(start_dim=2),
            nn.Dropout(0.1),
            nn.Linear(16 * 16, 768),  # 这里的7*7是根据resnet50，原img大小为224*224的情况来的
            nn.ReLU(inplace=True)
        )

        self.text_img_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=1,
            dropout=0.1,
            batch_first=True
        )
        self.img_text_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=1,
            dropout=0.1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 3)

    def forward(self, token_ids, segment_ids, imgs):
        # 文本
        hidden_states, pooling = self.bert([token_ids, segment_ids])

        # attention
        att_output = self.att(hidden_states)

        # 胶囊网络
        cap_output = self.caps_layer(att_output)

        # 图像
        imgs = imgs.permute(0, 3, 1, 2)
        img_output = self.resnet(imgs)
        img_output = self.hidden_trans(img_output)

        # img_text = self.img_text_attention(img_output, cap_output, cap_output)[0]
        text_img = self.text_img_attention(cap_output, img_output, img_output)[0]
        text_img = torch.mean(text_img, dim=1)
        # img_text = torch.mean(img_text, dim=1)
        # feature_output = torch.cat((text_img, img_text), dim=-1)
        text_img_drop = self.dropout(text_img)

        return text_img_drop, text_img


model = Model().to(device)


# 定义使用的loss和optimizer，这里支持自定义
class Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, targets):
        output, text_img_drop, text_img = outputs

        loss_cross = nn.CrossEntropyLoss()
        kl1_loss = F.kl_div(text_img_drop.softmax(dim=-1).log(), text_img.softmax(dim=-1), reduction='mean')
        kl2_loss = F.kl_div(text_img.softmax(dim=-1).log(), text_img_drop.softmax(dim=-1), reduction='mean')
        kl_loss = 0.5 * (kl1_loss + kl2_loss)
        cr_loss = loss_cross(output, targets)
        return kl_loss + cr_loss


model.compile(
    loss=Loss(),
    optimizer=optim.AdamW(model.parameters(), lr=1e-5),
    metrics=['accuracy']
)


# 定义评价函数
def evaluate(data):
    total, right = 0., 0.
    pre = []
    gro = []
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true)[0].argmax(axis=1)
        pre += y_pred.tolist()
        gro += y_true.tolist()
        total += len(y_true)
        right += (y_true == y_pred).sum().item()
    confusion = confusion_matrix(gro, pre)
    acc = right / total
    precision = precision_score(gro, pre, average='macro')
    recall = recall_score(gro, pre, average='macro')
    f1 = f1_score(gro, pre, average='macro')
    return confusion, acc, precision, recall, f1


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.
        self.best_val_pre = 0.
        self.best_val_recall = 0.
        self.best_val_f1 = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        confusion, acc, pre, recall, f1 = evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_acc = acc
            self.best_val_pre = pre
            self.best_val_recall = recall
            self.best_val_f1 = f1
            # model.save_weights('./best_model.pt')
        print(f'Epoch: {epoch}')
        print('混淆矩阵')
        print(confusion)
        print(f'acc: {acc:.5f}, pre: {pre:.5f}, recall: {recall:.5f}, f1: {f1:.5f}')
        print(f'best_val_acc: {self.best_val_acc:.5f}, best_val_pre: {self.best_val_pre:.5f},'
              f' best_val_recall: {self.best_val_recall:.5f}, best_val_f1: {self.best_val_f1:.5f}')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
    evaluate(valid_dataloader)
else:
    model.load_weights('best_model.pt')

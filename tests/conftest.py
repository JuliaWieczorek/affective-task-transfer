import pytest
import torch
from transformers import BertConfig,BertModel,BertTokenizerFast

@pytest.fixture(scope="session")
def backbone(tmp_path_factory):
    root=tmp_path_factory.mktemp("bert")
    (root/"vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\nfeel\nsad\n",encoding="utf-8")
    BertTokenizerFast(vocab_file=str(root/"vocab.txt")).save_pretrained(root)
    torch.manual_seed(4)
    BertModel(BertConfig(vocab_size=8,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)).save_pretrained(root)
    return str(root)

@pytest.fixture
def rows():
    return [{"id":str(i),"text":"i feel sad", "sentiment":i%3,"emotion":[1,i%2],"intensity":[i%3,1 if i%2 else -100],"group_id":str(i),"parent_id":str(i),"is_augmented":False} for i in range(6)]

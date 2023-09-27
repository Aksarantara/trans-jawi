from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io
import torch
import os
from pydantic import BaseModel
from jawi_models import *
import numpy as np

def load_pretrained(model, weight_path, device, flexible = False):
    if not weight_path:
        return model

    pretrain_dict = torch.load(weight_path) if device == 'cuda' else torch.load(weight_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    if flexible:
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    print("Pretrained layers:", pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model

DEVICE = 'cpu'
jawi_dim = 56
rumi_dim = 36
enc_emb_dim = 300
dec_emb_dim = 300
enc_hidden_dim = 512
dec_hidden_dim = 512
rnn_type = "lstm"
enc2dec_hid = True
attention = True
enc_layers = 1
dec_layers = 2
m_dropout = 0
enc_bidirect = True
enc_outstate_dim = enc_hidden_dim * (2 if enc_bidirect else 1)

# Jawi Malay-Rumi transliteration
enc_jawi = Encoder(
    input_dim=jawi_dim,
    embed_dim=enc_emb_dim,
    hidden_dim=enc_hidden_dim,
    rnn_type=rnn_type,
    layers=enc_layers,
    dropout=m_dropout,
    device=DEVICE,
    bidirectional=enc_bidirect,
)
dec_jawi = Decoder(
    output_dim=jawi_dim,
    embed_dim=dec_emb_dim,
    hidden_dim=dec_hidden_dim,
    rnn_type=rnn_type,
    layers=dec_layers,
    dropout=m_dropout,
    use_attention=attention,
    enc_outstate_dim=enc_outstate_dim,
    device=DEVICE,
)
enc_rumi = Encoder(
    input_dim=rumi_dim,
    embed_dim=enc_emb_dim,
    hidden_dim=enc_hidden_dim,
    rnn_type=rnn_type,
    layers=enc_layers,
    dropout=m_dropout,
    device=DEVICE,
    bidirectional=enc_bidirect,
)
dec_rumi = Decoder(
    output_dim=rumi_dim,
    embed_dim=dec_emb_dim,
    hidden_dim=dec_hidden_dim,
    rnn_type=rnn_type,
    layers=dec_layers,
    dropout=m_dropout,
    use_attention=attention,
    enc_outstate_dim=enc_outstate_dim,
    device=DEVICE,
)

jawi_glyph_path = "weights/jawi_glyph.pth"
rumi_glyph_path = "weights/rumi_glyph.pth"
pretrain_wgt_j2r_path = "weights/j2r_v2.pth"
pretrain_wgt_r2j_path = "weights/r2j_v2.pth"

jawi_glyph = torch.load(jawi_glyph_path)
rumi_glyph = torch.load(rumi_glyph_path)

model_j2r = Seq2Seq(enc_jawi, dec_rumi, pass_enc2dec_hid=enc2dec_hid, device=DEVICE)
model_r2j = Seq2Seq(enc_rumi, dec_jawi, pass_enc2dec_hid=enc2dec_hid, device=DEVICE)

model_j2r = load_pretrained(model_j2r, pretrain_wgt_j2r_path, device=DEVICE).to(DEVICE)
model_r2j = load_pretrained(model_r2j, pretrain_wgt_r2j_path, device=DEVICE).to(DEVICE)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def root():
    return {'hello': 'ready for jawi transliteration'}

@app.get("/ping")
async def ping():
    return {"status": "Healthy"}

def replace_kaf_gaf_unicode(word):
    word = word.replace('ک', 'ك')
    word = word.replace('ݢ', 'ڬ')
    return word

def inferencer(word, model, src_glyph, tgt_glyph, device=DEVICE, topk = 10):
    word = replace_kaf_gaf_unicode(word)

    in_vec = torch.from_numpy(src_glyph.word2xlitvec(word)).to(device)
    ## change to active or passive beam
    p_out_list = model.active_beam_inference(in_vec, beam_width = topk)
    p_result = [ tgt_glyph.xlitvec2word(out.cpu().numpy()) for out in p_out_list]

    result = p_result

    return result

def inference_looper(in_words, model, src_dict, tgt_dict, topk = 3):
    res = []
    for w in in_words:
        res.append(inferencer(w, model, src_dict, tgt_dict, topk=topk))
    return res

class Transliteration(BaseModel):
    text: str

# Endpoint to receive a POST request with text data
@app.post("/j2r")
async def j2r(body: Transliteration):
    text = body.text.split()
    res = inference_looper(text, model_j2r, jawi_glyph, rumi_glyph, topk = 1)
    res = ' '.join(sublist[0] for sublist in res)
    return {"rumi": res}

# Endpoint to receive a POST request with text data and language preference
@app.post("/r2j")
async def r2j(body: Transliteration):
    text = body.text.split()
    res = inference_looper(text, model_r2j, rumi_glyph, jawi_glyph, topk = 1)
    res = ' '.join(sublist[0] for sublist in res)
    return {"jawi": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

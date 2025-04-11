from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
import torch
from transformers import BertTokenizer
from src.features.tokenization import parallel_tokenize
from src.models.transformer import Transformer
from src.evaluation.model_evaluation import generate_summaries_transformer

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

with open("model/config_model.json", "r") as f:
    config = json.load(f)

modelTransformer = Transformer(
    pad_idx=config["pad_idx"],
    voc_size=config["voc_size"],
    hidden_size=config["hidden_size"],
    n_head=config["n_head"],
    max_len=config["max_len"],
    dec_max_len=config["dec_max_len"],
    ffn_hidden=config["ffn_hidden"],
    n_layers=config["n_layers"]
)

# Charger les poids
modelTransformer.load_state_dict(torch.load("model/pytorch_model.bin", map_location="cpu"))
modelTransformer.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelTransformer.to(device)

# Fonction pour obtenir le nombre de CPU disponibles
def get_allowed_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

cpu_count = get_allowed_cpu_count()
n_process = max(1, cpu_count // 2)

# Route pour afficher le formulaire
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": None})

# Route pour traiter le formulaire et générer le résumé
@app.post("/summarize/", response_class=HTMLResponse)
def summarize_text(request: Request, text: str = Form(...)):
    print("Texte reçu :", text)
    tokenized_input = parallel_tokenize(
        [text],
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=512,
    )
    summaries = generate_summaries_transformer(modelTransformer, batch_size=32, tokenized_input=tokenized_input)
    print(summaries[0])
    return templates.TemplateResponse("index.html", {"request": request, "summary": summaries[0]})


''' api works : in Terminal 

uvicorn app.api:app --reload
curl -X POST "http://127.0.0.1:8000/summarize/" -H "Content-Type: application/x-www-form-urlencoded" -d "text=new york police concerned drone tool terrorist investigate way stop potential attack police acknowledge drone potential weapon nypd say technology advance use carry air assault chemical weapon firearm police want develop technology allow control drone scan sky major event nypd say drone carry explosive number threat investigate way stop attack deputy chief salvatore dipace left concern incident year drone land german chancellor angela merkel take chancellor people drone fly pack football stadium manchester england week ago result suspect pilot arrest consult military member counterterrorism bomb squad emergency service aviation unit work plan counter weaponize drone nypd receive intelligence indicate imminent threat increasingly concerned year deputy chief salvatore dipace tell cbs news look people jury rig drone carry gun carry different type explosive want possibility worried mr dipace say police see video show accurate attack drone see video drone fly different target route accurately hit target paintball nypd see drone carry explosive number threat mr dipace concern follow incident germany year drone able land german chancellor angela merkel deliver speech drone circle land ms merkel deliver speech sin germany spark fear device easily commit terrorist act say think happen drone hit target right mark take chancellor people dramatic increase incident involve drone new york city year 40 record case unmanned aircraft system drone fly airspace nypd helicopter incident summer drone 800 foot ground nearly collide police helicopter nypd aviation unit member sergeant antonio hernandez say fly dark night vision goggle try job thing know drone come altitude"

'''
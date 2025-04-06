from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
from transformers import BertTokenizer
from src.features.tokenization import parallel_tokenize
from src.models.transformer import Transformer
from src.evaluation.model_evaluation import generate_summaries_transformer

def get_allowed_cpu_count():
    # Returns the number of CPU cores available for this process.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

cpu_count = get_allowed_cpu_count()
print(cpu_count)
n_process = max(1, cpu_count // 2)

app = FastAPI()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize the model
modelTransformer = Transformer(
    pad_idx=0,
    voc_size=tokenizer.vocab_size,
    hidden_size=128,
    n_head=8,
    max_len=512,
    dec_max_len=128,
    ffn_hidden=128,
    n_layers=3,
)

# Load the model weights
modelTransformer.load_state_dict(torch.load("model_weights/transformer_weights_25_epochs.pth"))
modelTransformer.eval()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelTransformer.to(device)

class SummaryRequest(BaseModel):
    articles: list

@app.post("/generate_summary/")
async def generate_summary(request: SummaryRequest):
    try:
        # Tokenize the input texts
        tokenized_input = parallel_tokenize(
            request.articles,
            tokenizer_name="bert-base-uncased",
            max_workers=n_process,
            chunk_size=2000,
            max_length=512,
        )

        # Generate summaries
        summaries = generate_summaries_transformer(modelTransformer, batch_size=32, tokenized_input=tokenized_input)

        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api works : curl -X POST "http://127.0.0.1:8000/generate_summary/" -H "Content-Type: application/json" -d '{"articles": ["new york police concerned drone tool terrorist investigate way stop potential attack police acknowledge drone potential weapon nypd say technology advance use carry air assault chemical weapon firearm police want develop technology allow control drone scan sky major event nypd say drone carry explosive number threat investigate way stop attack deputy chief salvatore dipace left concern incident year drone land german chancellor angela merkel take chancellor people drone fly pack football stadium manchester england week ago result suspect pilot arrest consult military member counterterrorism bomb squad emergency service aviation unit work plan counter weaponize drone nypd receive intelligence indicate imminent threat increasingly concerned year deputy chief salvatore dipace tell cbs news look people jury rig drone carry gun carry different type explosive want possibility worried mr dipace say police see video show accurate attack drone see video drone fly different target route accurately hit target paintball nypd see drone carry explosive number threat mr dipace concern follow incident germany year drone able land german chancellor angela merkel deliver speech drone circle land ms merkel deliver speech sin germany spark fear device easily commit terrorist act say think happen drone hit target right mark take chancellor people dramatic increase incident involve drone new york city year 40 record case unmanned aircraft system drone fly airspace nypd helicopter incident summer drone 800 foot ground nearly collide police helicopter nypd aviation unit member sergeant antonio hernandez say fly dark night vision goggle try job thing know drone come altitude"]}'

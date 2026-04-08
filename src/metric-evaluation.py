import torch
import json
import argparse
import os
import sacrebleu
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from datetime import datetime

# --- 0. Argomenti dello script ---
parser = argparse.ArgumentParser(description="Inference e Valutazione con Constrained Decoding")
parser.add_argument("--input_file", type=str, required=True, help="Path del file test_dict.json")
parser.add_argument("--direction", type=str, required=True, choices=["it_lij", "lij_it"], help="Direzione")
parser.add_argument("--setup", type=str, default=None, choices=["por", "fr"], help="Nome setup specifico")
parser.add_argument("--num_print", type=int, default=10, help="Esempi da stampare")
parser.add_argument("--save", action="store_true", default=False, help="Salva le predizioni complete")
args = parser.parse_args()

model_name = args.direction
if args.setup:
    model_name += f"_{args.setup}"

model_path = f"./model_mbart_{model_name}_final"

# --- 1. Caricamento Modello e Tokenizer ---
print(f"[INFO] Caricamento modello per valutazione: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path, 
    quantization_config=bnb_config, 
    device_map="auto",
    local_files_only=True
)

# --- 2. Logica per la Whitelist dei Token (Constrained Decoding) ---
def get_allowed_tokens(tokenizer, input_file):
    print("[INFO] Generazione whitelist dei token permessi...")
    allowed_ids = set()
    
    # Token speciali e di controllo
    special_tokens = [
        tokenizer.pad_token_id, 
        tokenizer.eos_token_id, 
        tokenizer.bos_token_id,
        tokenizer.unk_token_id,
        tokenizer.convert_tokens_to_ids("<lij>"),
        tokenizer.convert_tokens_to_ids("it_IT")
    ]
    allowed_ids.update([t for t in special_tokens if t is not None])

    # Estrazione token dal dataset (per coprire tutto il lessico ligure/italiano noto)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k, v in data.items():
            # Prendiamo i token da entrambe le lingue nel JSON per sicurezza
            toks = tokenizer.encode(v, add_special_tokens=False)
            allowed_ids.update(toks)
    
    return list(allowed_ids)

ALLOWED_TOKENS_LIST = get_allowed_tokens(tokenizer, args.input_file)

class LigurianConstraintProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask

# --- 3. Funzione di Traduzione ---
def translate(text, direction):
    if direction == "it_lij":
        tokenizer.src_lang = "it_IT"
        tgt_lang_token = "<lij>"
    else:
        tokenizer.src_lang = "<lij>"
        tgt_lang_token = "it_IT"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    
    processors = LogitsProcessorList([LigurianConstraintProcessor(ALLOWED_TOKENS_LIST)])

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_token),
        max_length=128,
        num_beams=4,
        early_stopping=True,
        logits_processor=processors
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# --- 4. Processo di Valutazione ---
with open(args.input_file, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

indices = sorted(list(set([k.split('_')[1] for k in input_data.keys() if '_' in k])), key=int)

predictions = []
references = []
results_to_save = {}

print(f"\n--- VALUTAZIONE IN CORSO ({len(indices)} frasi) ---")

for count, _ in enumerate(indices):
    if args.num_print and count >= args.num_print:
        break
    
    idx = random.choice(indices)
    if args.direction == "it_lij":
        src = input_data.get(f"x_{idx}")
        ref = input_data.get(f"y_{idx}")
    else:
        src = input_data.get(f"y_{idx}")
        ref = input_data.get(f"x_{idx}")

    if not src or not ref: continue
        
    pred = translate(src, args.direction)
    predictions.append(pred)
    references.append(ref)
    
    if args.save:
        results_to_save[f"x_{idx}"] = pred if args.direction == "lij_it" else src
        results_to_save[f"y_{idx}"] = src if args.direction == "lij_it" else pred

    if count < args.num_print:
        print(f"\n[ID {idx}]")
        print(f"  SRC: {src}")
        print(f"  REF: {ref}")
        print(f"  PRD: {pred}")

    
    if (count + 1) % 100 == 0:
        print(f"Avanzamento: {count + 1}/{len(indices)}...")

# --- 5. Calcolo Metriche ---
print("\n" + "="*50)
print("REPORT FINALE METRICHE")
print("="*50)

bleu = sacrebleu.corpus_bleu(predictions, [references])
spbleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='flores101')
chrf = sacrebleu.corpus_chrf(predictions, [references])

print(f"BLEU Score:  {bleu.score:.2f}")
print(f"spBLEU Score: {spbleu.score:.2f}")
print(f"chrF++ Score: {chrf.score:.2f}")
print("="*50)

# --- 6. Salvataggio ---
metrics_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "direction": args.direction,
    "bleu": round(bleu.score, 2),
    "spbleu": round(spbleu.score, 2),
    "chrf": round(chrf.score, 2)
}

metrics_path = f"metrics/metrics_{model_name}.json"
os.makedirs("metrics", exist_ok=True)
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, ensure_ascii=False, indent=2)

if args.save:
    output_path = f"predictions/eval_full_{model_name}.json"
    os.makedirs("predictions", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
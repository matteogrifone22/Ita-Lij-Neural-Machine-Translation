import json
import torch
import os
import argparse
import random
from datetime import datetime
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 0. Gestione Argomenti ---
parser = argparse.ArgumentParser(description="Script NMT a Due Fasi (Dizionario -> Frasi)")
parser.add_argument("--model_type", type=str, default="mbart", choices=["mbart"], help="Architettura modello")
parser.add_argument("--direction", type=str, required=True, choices=["it_lij", "lij_it"], help="Direzione")
parser.add_argument("--setup", type=str, default=None, choices=["por", "fr"], help="Nome setup specifico (es. mbart-lij-it-base-por) per inizializzazione pesi")
parser.add_argument("--skip_fase1", action="store_true", help="Salta il training sul dizionario")
parser.add_argument("--skip_fase2", action="store_true", help="Salta il training sulle frasi")
parser.add_argument("--epochs_f1", type=int, default=5, help="Epoche Fase 1 (Dizionario)")
parser.add_argument("--epochs_f2", type=int, default=20, help="Epoche Fase 2 (Frasi)")
args = parser.parse_args()


# --- 1. Configurazione Path ---
exp_id = f"{args.model_type}_{args.direction}"

base_model_path = f"model/mbart_{args.direction}_base" if args.model_type == "mbart" else "path/to/nllb"
if args.setup:
    base_model_path += f"_{args.setup}"
    exp_id += f"_{args.setup}"
    
print(f"\n[CONFIG] Modello: {args.model_type.upper()}, Direzione: {args.direction}, Setup: {args.setup or 'default'}")
# Path Dati
DICT_PATH = "preprocessed_data/TIG.json"
TRAIN_FRASI = "preprocessed_data/train_dict.json"
VAL_FRASI = "preprocessed_data/val_dict.json"
SYNTH_DATA = "preprocessed_data/backtranslated_data.json"

# Directory di output
OUTPUT_DIR_F1 = f"./results_{exp_id}_fase1"
OUTPUT_DIR_F2 = f"./results_{exp_id}_fase2"
FINAL_MODEL_DIR = f"./model_{exp_id}_final"

# --- 2. Funzioni di Supporto ---
def load_and_prepare_data(file_paths, sample_ratio=1.0):
    if isinstance(file_paths, str): file_paths = [file_paths]
    all_formatted = []

    for path in file_paths:
        if not path or not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        indices = set([k.split('_')[1] for k in data.keys() if '_' in k])
        current_formatted = []
        for idx in sorted(list(indices), key=lambda x: int(x) if x.isdigit() else x):
            it_sent, lij_sent = data.get(f"x_{idx}"), data.get(f"y_{idx}")
            if it_sent and lij_sent:
                src = it_sent if args.direction == "it_lij" else lij_sent
                tgt = lij_sent if args.direction == "it_lij" else it_sent
                current_formatted.append({"src": src, "tgt": tgt})
        
        # Applica sampling se richiesto (usato per il dizionario in Fase 2)
        if sample_ratio < 1.0:
            random.shuffle(current_formatted)
            current_formatted = current_formatted[:int(len(current_formatted) * sample_ratio)]
            
        all_formatted.extend(current_formatted)
        print(f"Caricate {len(current_formatted)} coppie da: {path}")

    return Dataset.from_list(all_formatted)

def preprocess_function(examples):
    if args.model_type == "mbart":
        tokenizer.src_lang = "it_IT" if args.direction == "it_lij" else "<lij>"
    inputs = tokenizer(examples["src"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["tgt"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

# --- 3. Inizializzazione Modello e Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
)

def get_model(path):
    m = AutoModelForSeq2SeqLM.from_pretrained(path, quantization_config=bnb_config, device_map="auto")
    m = prepare_model_for_kbit_training(m)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, task_type="SEQ_2_SEQ_LM"
    )
    return get_peft_model(m, lora_config)
# ==========================================
# FASE 1: TRAINING SUL DIZIONARIO (Lessico)
# ==========================================
model = get_model(base_model_path)

if not args.skip_fase1:
    print("\n>>> INIZIO FASE 1: DIZIONARIO COMPLETO")
    
    def preprocess_fase1(examples):
        if args.model_type == "mbart":
            tokenizer.src_lang = "it_IT" if args.direction == "it_lij" else "<lij>"
        inputs = tokenizer(examples["src"], max_length=32, truncation=True, padding="max_length")
        labels = tokenizer(text_target=examples["tgt"], max_length=32, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    dict_ds = load_and_prepare_data(DICT_PATH).map(preprocess_fase1, batched=True)
    
    args_f1 = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR_F1,
        learning_rate=5e-4,
        num_train_epochs=args.epochs_f1,
        per_device_train_batch_size=32, 
        gradient_accumulation_steps=1, 
        fp16=True,
        save_strategy="no",
        logging_steps=100,              
        report_to="none"
    )
    
    trainer_f1 = Seq2SeqTrainer(
        model=model, args=args_f1,
        train_dataset=dict_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    trainer_f1.train()
    print(">>> FASE 1 COMPLETATA.")

else:
    print("\n>>> FASE 1 SALTATA. Caricamento modello base per Fase 2...")
    model = get_model(base_model_path)

# ==========================================
# FASE 2: TRAINING SULLE FRASI (Sintassi)
# ==========================================
if not args.skip_fase2:
    print("\n>>> INIZIO FASE 2: FRASI + 15% DIZIONARIO")
    
    # Dataset misto: Frasi Reali + Backtranslation + 15% del Dizionario
    list_files = [TRAIN_FRASI]
    if args.direction == "it_lij":
        print(f"Backtranslation attivata per it->lij.")
        if os.path.exists(SYNTH_DATA): list_files.append(SYNTH_DATA)
    
    ds_frasi = load_and_prepare_data(list_files)
    #ds_dict_sample = load_and_prepare_data(DICT_PATH, sample_ratio=0)
    train_ds_f2 = concatenate_datasets(ds_frasi).shuffle().map(preprocess_function, batched=True)
    val_ds_f2 = load_and_prepare_data(VAL_FRASI).map(preprocess_function, batched=True)

    args_f2 = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR_F2,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4, 
        num_train_epochs=args.epochs_f2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    )

    trainer_f2 = Seq2SeqTrainer(
        model=model, args=args_f2,
        train_dataset=train_ds_f2, eval_dataset=val_ds_f2,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    trainer_f2.train()
    trainer_f2.save_model(FINAL_MODEL_DIR)
    print(f"\n[FINISH] Modello finale salvato in {FINAL_MODEL_DIR}")
else:
    if not args.skip_fase1:
        model.save_pretrained(FINAL_MODEL_DIR)
        print(f"\n[FINISH] Salvato solo modello Fase 1 in {FINAL_MODEL_DIR}")
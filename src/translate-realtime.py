import torch
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# --- 0. Gestione Argomenti ---
parser = argparse.ArgumentParser(description="Traduttore Real-time mBART")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device da usare (cuda o cpu)")
parser.add_argument("--direction", type=str, default="it_lij", choices=["it_lij", "lij_it"], help="Direzione traduzione")
args = parser.parse_args()

MODEL_PATH = f"./model_mbart_{args.direction}_final"

print(f"[INFO] Configurazione: Device={args.device.upper()}, Direzione={args.direction}")
print(f"[INFO] Caricamento modello da: {MODEL_PATH}...")

# --- 1. Caricamento Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# --- 2. Caricamento Modello Condizionale ---
if args.device == "cuda":
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )   
else:
   
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        local_files_only=True
    )

def translate_single(text, direction):
    # Impostazione lingue mBART
    if direction == "it_lij":
        tokenizer.src_lang = "it_IT"
        tgt_lang_token = "<lij>"
    else:
        tokenizer.src_lang = "<lij>"
        tgt_lang_token = "it_IT"

   
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(args.device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_token),
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# --- 3. Loop Interattivo ---
print("\n" + "="*50)
print(f"TRADUTTORE INTERATTIVO [{args.direction.upper()}] su {args.device.upper()}")
print("Scrivi la frase e premi Invio. Digita 'exit' per uscire.")
print("="*50)

try:
    while True:
        user_input = input(f"\n[{args.direction.split('_')[0].upper()}] > ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
            
        if not user_input.strip():
            continue

        prediction = translate_single(user_input, args.direction)
        
        print(f"[{args.direction.split('_')[1].upper()}] > {prediction}")

except KeyboardInterrupt:
    print("\n[INFO] Chiusura in corso...")
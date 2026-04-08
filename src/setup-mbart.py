import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

def setup_mbart_with_genoese():
    # --- 0. Gestione Argomenti ---
    parser = argparse.ArgumentParser(description="Setup mBART con inizializzazione pesi specifica")
    parser.add_argument("--lang", type=str, required=True, choices=["ita", "fra", "por"], 
                        help="Lingua da cui copiare i pesi iniziali per <lij>")
    parser.add_argument("--output_name", type=str, required=True, 
                        help="Nome della cartella di output (es. mbart-lij-it-base-por)")
    args = parser.parse_args()

    # Mappatura codici lingua mBART
    lang_map = {
        "ita": "it_IT",
        "fra": "fr_XX",
        "por": "pt_XX"
    }
    source_lang_token = lang_map[args.lang]

    print("=" * 60)
    print(f"Setting up mBART for Genoese (Lij)")
    print(f"Initializing <lij> weights from: {source_lang_token} ({args.lang})")
    print("=" * 60)
    
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    
    # 1. Configurazione Quantizzazione
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # 2. Caricamento Tokenizer
    print(f"\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Aggiunta Token <lij>
    print(f"\n[2/5] Adding <lij> token...")
    tokenizer.add_special_tokens({'additional_special_tokens': ["<lij>"]})
    
    # 4. Caricamento Modello in 4-bit
    print(f"\n[3/5] Loading model in 4-bit...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 5. Resize Embeddings
    print(f"\n[4/5] Resizing embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    
    with torch.no_grad():
        # Otteniamo l'ID della lingua sorgente scelta e del nuovo token ligure
        src_id = tokenizer.convert_tokens_to_ids(source_lang_token)
        lij_id = tokenizer.convert_tokens_to_ids("<lij>")
        
        if src_id == tokenizer.unk_token_id:
            raise ValueError(f"Errore: Il token {source_lang_token} non è stato trovato nel tokenizer!")

        # Copiamo i pesi dell'embedding
        input_embeddings = model.get_input_embeddings().weight
        input_embeddings[lij_id] = input_embeddings[src_id].clone()
        
        # Copiamo anche i pesi dell'output embedding
        output_embeddings = model.get_output_embeddings().weight
        output_embeddings[lij_id] = output_embeddings[src_id].clone()
        
        print(f"Successfully initialized <lij> weights from {source_lang_token}")

    # 6. Salvataggio
    output_dir = Path(f"model/{args.output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[5/5] Saving to {output_dir}...")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    print(f"\nSetup complete! Target directory: {output_dir}")
    return tokenizer, model

if __name__ == "__main__":
    setup_mbart_with_genoese()
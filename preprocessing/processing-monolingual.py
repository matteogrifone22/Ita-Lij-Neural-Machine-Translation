import os
import json
from pathlib import Path

def read_monolingual_data(monolingua_folder):
    """
    Legge file di testo monolingua (solo Ligure).
    Restituisce un dizionario con chiavi y_i (Ligure).
    Nota: Lasciamo x_i vuoto o lo omettiamo, 
    verrà riempito dal modello di backtranslation.
    """
    data_dict = {}
    i = 0
    
    # Controlla se la cartella esiste
    if not os.path.exists(monolingua_folder):
        raise FileNotFoundError(f"Cartella non trovata: {monolingua_folder}")
    
    # Elenca tutti i file .txt
    files = sorted([f for f in os.listdir(monolingua_folder) if f.endswith('.txt')])
    
    print(f"Trovati {len(files)} file monolingua nella cartella '{monolingua_folder}'")
    
    for filename in files:
        file_path = os.path.join(monolingua_folder, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Salta righe vuote o header
                    if line == '' or 'zeneixi' in line.lower():
                        continue
                    
                    # Salviamo solo la parte Ligure (y)
                    # In fase di backtranslation, il modello genererà la x (Italiano)
                    y_key = f"y_{i}"
                    data_dict[y_key] = line
                    i += 1
        except Exception as e:
            print(f"Errore nella lettura di {filename}: {e}")
    
    return data_dict

def save_monolingual_json(data_dict, output_path='preprocessed_data/monolingua_lij.json'):
    """
    Salva il dizionario monolingua in un unico file JSON.
    """
    # Crea la cartella di output se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset monolingua salvato in {output_path}")
    print(f"Totale frasi da tradurre: {len(data_dict)}")

if __name__ == "__main__":
    # Path della cartella dove tieni i testi solo in Ligure (es. Wikipedia, libri, ecc.)
    monolingual_input_path = 'dataset_lij' 
    
    # Esecuzione
    data_dict = read_monolingual_data(monolingual_input_path)
    
    if data_dict:
        save_monolingual_json(data_dict)
    else:
        print("Nessun dato trovato. Controlla il percorso o i file .txt")
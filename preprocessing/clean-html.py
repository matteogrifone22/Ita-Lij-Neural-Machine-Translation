import os
import glob
import json
import re
from bs4 import BeautifulSoup

def clean_text(text):
    """Pulisce il testo gestendo gli apostrofi tipografici e artefatti di codifica"""
    if not text:
        return ""
        
    # 1. Correzione manuale dei "glitch" di codifica comuni
    text = text.replace('\x92', "'")  # Codice esadecimale per l'apostrofo storto
    text = text.replace('’', "'")     
    text = text.replace('’', "'")     
    text = text.replace('‘', "'")     
    #rimuoviamo i numeri 
    text = re.sub(r'/\d+', '', text)
    
    # Altre correzioni possibili per caratteri "liguri" che saltano spesso
    text = text.replace('—', "—")     
    
    # 2. Rimuove i codici tra parentesi quadre
    text = re.sub(r'\[.*?\]', '', text)
    
    # 3. Rimuove note tra parentesi tonde
    text = re.sub(r'\(.*?\)', '', text)
    
    # 4. Normalizzazione spazi e minuscole
    text = " ".join(text.split())
    return text.lower().strip()

def process_dictionary_files(input_folder, output_json):
    html_files = glob.glob(os.path.join(input_folder, "*.html"))
    all_data = {}
    global_counter = 0
    seen_pairs = set() 

    for file_path in html_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        for p in soup.find_all('p'):
            ita_tag = p.find('font', class_='ita')
            zen_tag = p.find('font', class_='zen')
            
            if ita_tag and zen_tag:
                # 1. Saltiamo quelli con il link <a>
                if zen_tag.find('a'):
                    continue
                
                # 2. Estraiamo il testo grezzo
                ita_raw = ita_tag.get_text()
                zen_raw = zen_tag.get_text()
                
                # 3. Logica Sinonimi vs Perifrasi:
                # Se c'è la virgola, sono sinonimi -> prendo il primo.
                # Se non c'è la virgola (come "ch’o tîa a-o bleu"), prendo tutto.
                if ',' in zen_raw:
                    zen_final = zen_raw.split(',')[0]
                else:
                    zen_final = zen_raw
                
                # 4. Pulizia finale
                ita_clean = clean_text(ita_raw)
                zen_clean = clean_text(zen_final)
              
                
                if ita_clean and zen_clean and "vedi" not in zen_clean:
                    if (ita_clean, zen_clean) not in seen_pairs:
                        all_data[f"x_{global_counter}"] = ita_clean
                        all_data[f"y_{global_counter}"] = zen_clean
                        seen_pairs.add((ita_clean, zen_clean))
                        global_counter += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    print(f"Completato! Estratti {len(all_data)} lemmi/espressioni.")

# Esecuzione
process_dictionary_files('html/', 'preprocessed_data/TIG.json')
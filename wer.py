#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import unicodedata

import pandas as pd
import jiwer
from pydub import AudioSegment

CSV_PATH = "result.csv"
DATASET_PATH = "dataset.json"

DIGIT_TO_WORD = {
    "0": "zero",
    "1": "un",
    "2": "deux",
    "3": "trois",
    "4": "quatre",
    "5": "cinq",
    "6": "six",
    "7": "sept",
    "8": "huit",
    "9": "neuf",
}

def normalize_for_wer(s: str) -> str:
    """Normalise le texte pour le calcul WER : lowercase, sans accents, seulement alphabet, sans espaces.
    Retourne une chaîne avec des espaces entre chaque caractère pour permettre la comparaison par blocs."""
    if s is None:
        return ""
    s = str(s)
    # remplacer les chiffres par leur équivalent en lettres
    s = re.sub(r"\d", lambda m: DIGIT_TO_WORD.get(m.group(0), m.group(0)), s)
    # Normaliser les caractères Unicode (NFD) pour séparer les accents
    s = unicodedata.normalize('NFD', s)
    # Enlever les accents (caractères de type Mark)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("M"))
    # Mettre en lowercase
    s = s.lower()
    # Garder seulement l'alphabet, enlever tout le reste
    s = re.sub(r"[^a-z]", "", s)
    # Ajouter des espaces entre chaque caractère pour que jiwer.wer compare caractère par caractère
    return " ".join(s)

def compute_wer(truth_raw: str, hyp_raw: str) -> float:
    """Calcule le WER avec normalisation propre"""
    truth_normalized = normalize_for_wer(truth_raw)
    hyp_normalized = normalize_for_wer(hyp_raw)
    
    if truth_normalized == "" and hyp_normalized == "":
        return 0.0
    if truth_normalized == "" and hyp_normalized != "":
        return 1.0
    
    # Utiliser jiwer.wer directement
    value = jiwer.wer(truth_normalized, hyp_normalized)
    return min(1.0, float(value))

def pick_col(df, candidates):
    cols = list(df.columns)
    low = {c.lower().strip(): c for c in cols}
    for name in candidates:
        if name in low:
            return low[name]
    for c in cols:
        cl = c.lower()
        if any(name in cl for name in candidates):
            return c
    return None

def load_dataset_map(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    # Handle the structure with "data" array
    if "data" in data:
        for item in data["data"]:
            audio_file = item.get("audio_file") or item.get("audio_path", "")
            transcription = item.get("transcription") or item.get("text", "")
            if audio_file:
                base = os.path.basename(audio_file)
                base_l = base.lower()
                noext = os.path.splitext(base_l)[0]
                m[base_l] = transcription
                m[noext] = transcription
    else:
        # Fallback for other structures
        for k, v in data.items():
            base = os.path.basename(k)
            base_l = base.lower()
            noext = os.path.splitext(base_l)[0]
            m[base_l] = v
            m[noext] = v
    return m

def _sanitize_display(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        if not os.path.exists(audio_path):
            return 0.0
        
        # Try different audio formats
        try:
            audio = AudioSegment.from_mp3(audio_path)
        except:
            try:
                audio = AudioSegment.from_wav(audio_path)
            except:
                try:
                    audio = AudioSegment.from_file(audio_path)
                except:
                    return 0.0
        
        return len(audio) / 1000.0  # pydub returns milliseconds, convert to seconds
    except:
        return 0.0

def main():
    csv_path = "result.csv"  # Utiliser le fichier de résultats STT
    dataset_path = "dataset.json"  # Utiliser notre dataset.json

    df = pd.read_csv(csv_path, encoding='utf-8')
    dataset_map = load_dataset_map(dataset_path)

    audio_col = pick_col(df, ["sample","audio","file","path","audio_path","filepath","wav","wav_path","utterance"])
    stt_col   = pick_col(df, ["stt","model","engine","provider","system","asr"])
    hyp_col   = pick_col(df, ["transcription","text","transcript","prediction","hypothesis","output","result","recognized_text","asr_text"])

    if hyp_col is None:
        print("colonne de transcription introuvable.")
        sys.exit(1)

    ref_col = pick_col(df, ["reference","ref","ground_truth","target","gold","truth","expected"])
    response_time_col = pick_col(df, ["response_time","latency","time","duration"])
    
    # Set to None to allow all models, or set to specific models to filter
    # allowed_models = {"gemini-2.0-flash", "groq-whisper-large-v3", "save_test_py", "elevenlabs", "deepgram-nova-3"}
    allowed_models = None  # uncomment to see all models
    
    wers = []
    stts = []
    hyps = []
    response_times = []
    audio_durations = []
    
    # Group results by reference text for better display
    results_by_ref = {}
    first_non_empty_printed = False

    for _, row in df.iterrows():
        hyp = str(row[hyp_col]) if hasattr(row, '__getitem__') and pd.notna(row[hyp_col]) else ""
        if ref_col is not None and ref_col in row:
            ref = "" if pd.isna(row[ref_col]) else str(row[ref_col])
        else:
            if audio_col is not None and pd.notna(row[audio_col]):
                base = os.path.basename(str(row[audio_col]))
                key = os.path.splitext(base.lower())[0]
                ref = dataset_map.get(key) or dataset_map.get(base.lower()) or ""
                ref = ref.get("text") if isinstance(ref, dict) else ref
            else:
                ref = ""

        # treat raw-empty lines as separators: print blank line and skip scoring
        if (ref.strip() == "") and (hyp.strip() == ""):
            if first_non_empty_printed:
                print("")
            continue
            
        stt_name = str(row[stt_col]) if stt_col and pd.notna(row[stt_col]) else "unknown"
        
        # skip models not in allowed list (if filtering is enabled)
        if allowed_models is not None and stt_name not in allowed_models:
            continue

        # skip if no reference found in dataset
        if ref.strip() == "":
            continue

        w = compute_wer(ref, hyp)
        wers.append(w)
        stts.append(stt_name)
        hyps.append(hyp.strip())
        
        # collect response times
        if response_time_col and pd.notna(row[response_time_col]):
            try:
                rt = float(row[response_time_col])
                response_times.append(rt)
            except (ValueError, TypeError):
                pass
        
        # collect audio duration
        if audio_col and pd.notna(row[audio_col]):
            audio_filename = str(row[audio_col])
            # Construct full path to audio file
            audio_path = os.path.join("Audio", audio_filename)
            duration = get_audio_duration(audio_path)
            audio_durations.append(duration)
        
        # Group by reference text + audio type (_raw or _aug)
        disp_ref = _sanitize_display(ref)
        disp_hyp = _sanitize_display(hyp)
        
        # Determine audio type from filename
        audio_type = ""
        if audio_col is not None and pd.notna(row[audio_col]):
            audio_filename = str(row[audio_col])
            if "_raw" in audio_filename.lower():
                audio_type = "_raw"
            elif "_aug" in audio_filename.lower():
                audio_type = "_aug"
        
        # Create unique key combining ref text and audio type
        ref_key = f"{disp_ref}{audio_type}"
        
        if ref_key not in results_by_ref:
            results_by_ref[ref_key] = []
        results_by_ref[ref_key].append((stt_name, disp_hyp, w))
    
    # Find STTs with at least one non-empty hyp
    stt_has_non_empty = {}
    for i, stt in enumerate(stts):
        if stt not in stt_has_non_empty:
            stt_has_non_empty[stt] = False
        if hyps[i] != "":
            stt_has_non_empty[stt] = True
    valid_stts = {stt for stt, has_non_empty in stt_has_non_empty.items() if has_non_empty}
    
    # Display results grouped by reference
    for ref_key, results in results_by_ref.items():
        # NOTE: On garde TOUS les échantillons, même si les WER sont identiques
        # (commenté le filtre qui sautait les WER identiques)
        # wer_scores = [wer_score for _, _, wer_score in results]
        # if len(set(wer_scores)) == 1:
        #     continue
            
        if not first_non_empty_printed:
            first_non_empty_printed = True
        else:
            print()
        
        # Extract ref text and audio type from key
        if ref_key.endswith("_raw"):
            ref_text = ref_key[:-4]
            audio_type_label = " (SIMPLE)"
        elif ref_key.endswith("_aug"):
            ref_text = ref_key[:-4]
            audio_type_label = " (HARD)"
        else:
            ref_text = ref_key
            audio_type_label = ""
            
        print(f"resultat attendu : \"{ref_text}\"{audio_type_label}")
        for stt_name, hyp_text, wer_score in results:
            if stt_name in valid_stts:
                print(f"{stt_name} = \"{hyp_text}\" -> {wer_score:.3f}")

    # create dataframe with all data including response times
    data_dict = {"stt": stts, "wer": wers, "hyp": hyps}
    if len(response_times) == len(stts):
        data_dict["response_time"] = response_times
    else:
        # pad with NaN if lengths don't match
        rt_padded = response_times + [float('nan')] * (len(stts) - len(response_times))
        data_dict["response_time"] = rt_padded[:len(stts)]
    
    out = pd.DataFrame(data_dict)
    
    # Filter out STTs where all hyp are empty
    out = out[out["stt"].isin(valid_stts)]
    out = out.drop(columns=["hyp"])
    
    if out["stt"].nunique() > 1 or (out["stt"].nunique()==1 and out["stt"].iloc[0] != "unknown"):
        print("\naverage WER by stt:")
        grp = out.groupby("stt", dropna=False)["wer"].mean().sort_values()
        cnt = out.groupby("stt", dropna=False)["wer"].size()
        best_wer = grp.iloc[0]  # le meilleur (plus petit WER)
        
        for name, mean_wer in grp.items():
            if mean_wer == best_wer:
                print(f"{name} -> {mean_wer:.3f}")
            else:
                if best_wer > 0:
                    diff_pct = ((mean_wer - best_wer) / best_wer) * 100
                    print(f"{name} -> {mean_wer:.3f} (-{diff_pct:.2f}%)")
                else:
                    print(f"{name} -> {mean_wer:.3f}")
            
        # calcul WER=0 et WER=1 par STT
        print("\nnumber of WER=0 and WER=1 by stt:")
        for name in grp.index:
            stt_wers = out[out["stt"] == name]["wer"]
            wer_0_count = (stt_wers == 0.0).sum()
            wer_1_count = (stt_wers == 1.0).sum()
            print(f"{name} -> WER=0: {wer_0_count}, WER=1: {wer_1_count}")
            
        # calcul latence moyenne par STT
        if "response_time" in out.columns:
            print("\naverage latency by stt:")
            rt_grp = out.groupby("stt", dropna=False)["response_time"].mean().sort_values()
            rt_cnt = out.groupby("stt", dropna=False)["response_time"].count()
            best_rt = rt_grp.iloc[0]  # le meilleur (plus rapide)
            
            for name, mean_rt in rt_grp.items():
                if name in rt_cnt and rt_cnt[name] > 0:
                    if mean_rt == best_rt:
                        print(f"{name} -> {mean_rt:.3f}s")
                    else:
                        if best_rt > 0:
                            diff_pct = ((mean_rt - best_rt) / best_rt) * 100
                            print(f"{name} -> {mean_rt:.3f}s (+{diff_pct:.2f}%)")
                        else:
                            print(f"{name} -> {mean_rt:.3f}s")
    else:
        overall = float(out["wer"].mean()) if len(out) > 0 else float('nan')
        print(f"\naverage WER -> {overall:.3f}")
        
        # calcul WER=0 et WER=1 global
        if len(out) > 0:
            wer_0_count = (out["wer"] == 0.0).sum()
            wer_1_count = (out["wer"] == 1.0).sum()
            print(f"WER=0: {wer_0_count}, WER=1: {wer_1_count}")
            
        # calcul latence moyenne global
        if "response_time" in out.columns and len(out) > 0:
            avg_latency = out["response_time"].mean()
            print(f"average latency -> {avg_latency:.3f}s")
    
    # calcul latence moyenne globale si pas déjà affiché
    if out["stt"].nunique() > 1 and "response_time" in out.columns and len(out) > 0:
        avg_latency = out["response_time"].mean()
        print(f"\nglobal average latency -> {avg_latency:.3f}s")
    
    # calcul nombre total d'audios uniques testés et durée totale
    unique_audios = set()
    audio_duration_map = {}
    
    # recalcul pour avoir les vrais audios uniques
    for _, row in df.iterrows():
        stt_name = str(row[stt_col]) if stt_col and pd.notna(row[stt_col]) else "unknown"
        
        # skip models not in allowed list (if filtering is enabled)
        if allowed_models is not None and stt_name not in allowed_models:
            continue
            
        if audio_col and pd.notna(row[audio_col]):
            audio_filename = str(row[audio_col])
            unique_audios.add(audio_filename)
            
            # calculate duration only once per unique audio
            if audio_filename not in audio_duration_map:
                audio_path = os.path.join("Audio", audio_filename)
                duration = get_audio_duration(audio_path)
                audio_duration_map[audio_filename] = duration
    
    total_audios = len(unique_audios)
    if total_audios > 0:
        print(f"\ntotal number of audios tested -> {total_audios}")
        
        # calcul durée totale réelle
        if audio_duration_map:
            total_duration_seconds = sum(audio_duration_map.values())
            total_hours = total_duration_seconds / 3600
            total_minutes = (total_duration_seconds % 3600) / 60
            
            if total_hours >= 1:
                print(f"total duration -> {total_hours:.1f}h ({total_duration_seconds:.1f}s)")
            else:
                print(f"total duration -> {total_minutes:.1f}min ({total_duration_seconds:.1f}s)")

class TeeOutput:
    """Write to both stdout and a file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

if __name__ == "__main__":
    # Redirect output to both terminal and file
    output_file = "wer_results.txt"
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        main()
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nRésultats sauvegardés dans {output_file}")

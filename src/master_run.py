import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob

# Importam functiile tale existente din proiect
try:
    from loader import load_image
    from preprocessing import preprocess_image
    from segmentation import segment_organ_classical, method_list
    from postprocessing import postprocess_mask
    from visualization import display_results  # Folosit pentru a salva vizualizarea cu 3 panouri
except ImportError as e:
    print("EROARE CRITICA: Acest script trebuie rulat din folderul 'src' al proiectului!")
    print(f"Detalii eroare: {e}")
    exit()

# CONFIGURARE CAI
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Un nivel mai sus de src

# Foldere de date
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'input')
OUTPUT_VIS_DIR = os.path.join(PROJECT_ROOT, 'data', 'output_vizualizari')  # Aici punem imaginile cu 3 panouri
OUTPUT_MASKS_DIR = os.path.join(PROJECT_ROOT, 'data', 'output_masks')  # Aici punem doar mastile alb-negru (pt statistica)
GT_DIR = os.path.join(PROJECT_ROOT, 'data', 'ground_truth')
POSTER_DIR = os.path.join(PROJECT_ROOT, 'poster_graphics')

# Creare foldere daca nu exista
for d in [OUTPUT_VIS_DIR, OUTPUT_MASKS_DIR, GT_DIR, POSTER_DIR]:
    os.makedirs(d, exist_ok=True)

# PASUL 1: GENERARE REZULTATE (BATCH)
def run_batch_generation():
    print("\nPASUL 1: GENERARE SEGMENTARI PE TOATE IMAGINILE")
    print("-" * 60)
    
    # Cautam imagini
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not image_files:
        print(f"Nu am gasit imagini in {INPUT_DIR}")
        return False
    
    print(f"Am gasit {len(image_files)} imagini de procesat.")
    
    timings = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"\nProcessing: {img_name}...")
        
        # 1. Load
        try:
            original = load_image(img_path)
        except:
            print(f"  Skipping {img_name} (load error)")
            continue
        
        # 2. Preprocess
        processed, _, original_shape = preprocess_image(original)
        
        for method in method_list:
            # 3. Segmentare + Cronometrare
            start_time = time.time()
            try:
                raw_mask = segment_organ_classical(processed, method=method)
                
                # 4. Post-procesare
                final_mask = postprocess_mask(raw_mask, original_shape)
                duration = time.time() - start_time
            except Exception as e:
                print(f"Eroare la metoda {method}: {e}")
                continue
            
            # Salvam timpul
            timings.append({
                "Imagine": img_name,
                "Metoda": method,
                "Timp Executie (s)": duration
            })
            
            # A. Salvam Masca Curata (Alb-Negru) pentru statistici
            mask_filename = f"{img_name}_{method}.png"
            cv2.imwrite(os.path.join(OUTPUT_MASKS_DIR, mask_filename), final_mask)
            
            # B. Salvam Vizualizarea (3 panouri) pentru poster
            vis_filename = f"VIS_{img_name}_{method}.png"
            display_results(original, final_mask, save_path=os.path.join(OUTPUT_VIS_DIR, vis_filename))
            
            print(f"  {method}: {duration:.4f}s")
    
    # Salvam timpii
    df_time = pd.DataFrame(timings)
    df_time.to_csv(os.path.join(PROJECT_ROOT, "rezultate_timpi.csv"), index=False)
    print("\nGenerare completa! Mastile sunt in 'data/output_masks'.")
    return True

# PASUL 2: GENERARE GROUND TRUTH (CONSENS)
def generate_consensus_gt():
    print("\nPASUL 2: GENERARE GROUND TRUTH (VOT MAJORITAR)")
    print("-" * 60)
    
    # Luam lista de imagini unice procesate
    mask_files = os.listdir(OUTPUT_MASKS_DIR)
    
    # Extragem numele imaginilor originale
    unique_images = set()
    for f in mask_files:
        for m in method_list:
            if m in f:
                # Reconstruim numele imaginii originale
                img_name = f.replace(f"_{m}.png", "")
                unique_images.add(img_name)
                break
    
    print(f"Generez GT pentru {len(unique_images)} imagini...")
    
    for img_name in unique_images:
        masks = []
        
        # Colectam mastile pentru aceasta imagine de la toate metodele
        for method in method_list:
            mask_path = os.path.join(OUTPUT_MASKS_DIR, f"{img_name}_{method}.png")
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    masks.append(m)
        
        if not masks:
            continue
        
        # Algoritmul de vot
        h, w = masks[0].shape
        vote_grid = np.zeros((h, w), dtype=int)
        
        for m in masks:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            vote_grid += (m > 127).astype(int)
        
        # Majoritate: Daca > jumatate din metode zic "alb", e alb
        threshold = len(masks) // 2
        gt_mask = (vote_grid > threshold).astype(np.uint8) * 255
        
        # Salvare
        cv2.imwrite(os.path.join(GT_DIR, f"GT_{img_name}.png"), gt_mask)
        print(f"  Generated GT for: {img_name}")
    
    print("Ground Truth generat in 'data/ground_truth'.")

# PASUL 3: CALCUL STATISTICI
def calculate_final_stats():
    print("\nPASUL 3: CALCUL METRICI (REAL)")
    print("-" * 60)
    
    results = []
    
    # Iteram prin fisierele GT
    gt_files = glob.glob(os.path.join(GT_DIR, "GT_*.png"))
    
    for gt_path in gt_files:
        gt_filename = os.path.basename(gt_path)
        # Extragem numele imaginii originale: GT_test1.jpg.png -> test1.jpg
        original_name = gt_filename.replace("GT_", "").replace(".png", "")
        
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        _, gt_bin = cv2.threshold(gt_img, 127, 1, cv2.THRESH_BINARY)  # 0 si 1
        
        for method in method_list:
            # Cautam masca metodei
            mask_name = f"{original_name}_{method}.png"
            mask_path = os.path.join(OUTPUT_MASKS_DIR, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            pred_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if pred_img is None:
                continue
            
            # Resize daca e necesar
            if pred_img.shape != gt_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            _, pred_bin = cv2.threshold(pred_img, 127, 1, cv2.THRESH_BINARY)
            
            # Calcul Matrice Confuzie
            TP = np.sum((pred_bin == 1) & (gt_bin == 1))
            FP = np.sum((pred_bin == 1) & (gt_bin == 0))
            FN = np.sum((pred_bin == 0) & (gt_bin == 1))
            TN = np.sum((pred_bin == 0) & (gt_bin == 0))
            
            eps = 1e-7
            dice = (2 * TP) / (2 * TP + FP + FN + eps)
            acc = (TP + TN) / (TP + TN + FP + FN + eps)
            prec = TP / (TP + FP + eps)
            rec = TP / (TP + FN + eps)
            
            results.append({
                "Imagine": original_name,
                "Metoda": method,
                "Accuracy": acc * 100,
                "Precision": prec * 100,
                "Recall": rec * 100,
                "Dice Score": dice
            })
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(PROJECT_ROOT, "rezultate_metrici_finale.csv")
    df.to_csv(csv_path, index=False)
    
    # Agregare medii
    df_avg = df.groupby("Metoda")[["Accuracy", "Precision", "Recall", "Dice Score"]].mean()
    print("\nREZULTATE MEDII:")
    print(df_avg)
    df_avg.to_csv(os.path.join(PROJECT_ROOT, "rezultate_medii.csv"))
    
    return df_avg

# PASUL 4: GRAFICE
def generate_graphs(df_avg):
    print("\nPASUL 4: GENERARE GRAFICE")
    print("-" * 60)
    
    # 1. Grafic Dice
    plt.figure(figsize=(10, 6))
    dice_scores = df_avg['Dice Score'].sort_values()
    colors = ['red' if x < 0.8 else 'orange' if x < 0.9 else 'green' for x in dice_scores]
    bars = plt.barh(dice_scores.index, dice_scores.values, color=colors)
    plt.title("Performanta Medie (Dice Score)", fontsize=14, fontweight='bold')
    plt.xlabel("Dice Score")
    plt.xlim(0, 1.1)
    
    for i, v in enumerate(dice_scores.values):
        plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(POSTER_DIR, "grafic_dice_final.png"), dpi=300)
    print("Grafic Dice salvat.")

# MAIN
if __name__ == "__main__":
    # 1. Generam tot
    success = run_batch_generation()
    
    if success:
        # 2. Generam adevarul
        generate_consensus_gt()
        
        # 3. Calculam cifrele
        df_avg = calculate_final_stats()
        
        # 4. Desenam
        generate_graphs(df_avg)
        
        print("\nGATA! Tot proiectul a fost re-procesat si recalculat.")
        print(f"Verifica folderul: {POSTER_DIR}")
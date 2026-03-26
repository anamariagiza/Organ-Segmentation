import os
from loader import load_image
from preprocessing import preprocess_image
from segmentation import segment_organ_classical
from postprocessing import postprocess_mask
from visualization import display_results


def organ_segmentation_pipeline(image_path,output_folder, method,GUI=None):
    """
    Pipeline complet de segmentare a organelor 
    
    Args:
        image_path: Calea către imaginea de intrare
        method: Metoda de segmentare ('adaptive_threshold', 'otsu', 'watershed')
        save_result: Salveaza rezultatul final
    """
    print("="*70)
    print("SISTEM DE SEGMENTARE A ORGANELOR")
    print("="*70)
    
    # 1. Încărcarea imaginii
    print("\n[1/5] Încărcare imagine...")
    GUI.modify_status("[1/5] Loading Image...")
    original_image = load_image(image_path)
    if original_image is None:
        return None, None
    
    # 2. Preprocesare
    print("\n[2/5] Preprocesare...")
    GUI.modify_status("[2/5] Preprocessing...")
    processed_image, normalized, original_shape = preprocess_image(original_image)
    
    # 3. Segmentare
    print(f"\n[3/5] Segmentare (metodă: {method})...")
    GUI.modify_status(f"[3/5] Segmentation (method: {method})...")
    mask = segment_organ_classical(processed_image, method=method)
    
    # 4. Post-procesare
    print("\n[4/5] Post-procesare...")
    GUI.modify_status("[4/5] Postprocessing...")
    final_mask = postprocess_mask(mask, original_shape,int(GUI.organ_no_box.get()))
    
    # 5. Afișarea rezultatului
    print("\n[5/5] Afisare rezultat...")
    GUI.modify_status("[5/5] Displaying result...")
    image_name = os.path.basename(image_path)
    output_image_name = f'{image_name}_segmentare_{method}.png' 
    save_path=os.path.join(output_folder,output_image_name) if os.path.exists(output_folder) else None
    display_results(original_image, final_mask, save_path)
    
    print("\n" + "="*70)
    print("SEGMENTARE COMPLETA!")
    if GUI:
        GUI.modify_status("Segmentation completed!")
    print("="*70)
    
    return final_mask, processed_image


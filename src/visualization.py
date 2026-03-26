import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def display_results(original_image, final_mask, save_path=None):
    """
    Afișează rezultatul: imaginea originala + masca suprapusa
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imaginea originală
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Imaginea Originala', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Masca de segmentare
    axes[1].imshow(final_mask, cmap='jet')
    axes[1].set_title('Masca de Segmentare', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Suprapunere
    overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Crearea mastii colorate
    colored_mask = np.zeros_like(overlay)
    colored_mask[:, :, 0] = final_mask  # Canal rosu
    
    # Suprapunem cu transparenta
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
    
    # Adaugam contur
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Suprapunere + Contur', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    # Salveaza rezultatul
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rezultat salvat la: {save_path}")
        
        base_path = os.path.splitext(save_path)[0]
        ext = os.path.splitext(save_path)[1]
        mask_path = f"{base_path}_mask{ext}"
        
        cv2.imwrite(mask_path, final_mask)
        print(f" Masca black & white salvată la: {mask_path}")
    
    plt.show()
    print("Afisare completa")



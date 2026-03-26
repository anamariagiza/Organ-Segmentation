import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from segmentation import method_list
from pipeline import organ_segmentation_pipeline
import os 

class AppGUI(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        self.organ_numbers = [1,2]
        
        # ---- Window Setup ----
        parent.title("Organ Segmentation App")
        parent.geometry("400x400")
        parent.resizable(False, False)

        # Configure main frame grid
        self.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)

        # ---- Title Frame ----
        title_frame = tk.Frame(self)
        title_frame.grid(row=0, column=0, sticky="ew", pady=10)

        title_label = tk.Label(
            title_frame,
            text="⚕️ Organ Segmentation App",
            fg="black",
            font=("Arial", 15, "bold")
        )
        title_label.pack()

        # ---- Settings Frame ----
        settings_frame = tk.Frame(self)
        settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        # Input directory
        tk.Label(settings_frame, text="Input Image Path:").grid(row=0, column=0, sticky="w", pady=5)
        self.input_entry = tk.Entry(settings_frame,width=25)
        self.input_entry.grid(row=0, column=1, pady=5)
        tk.Button(settings_frame, text="Browse",command=self.browse_for_image).grid(row=0, column=2, padx=5)

        # Output directory
        tk.Label(settings_frame, text="Output Image Path:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_entry = tk.Entry(settings_frame,width=25)
        self.output_entry.grid(row=1, column=1, pady=5)
        tk.Button(settings_frame, text="Browse",command=self.browse_for_folder).grid(row=1, column=2, padx=5)

        # Method selector
        tk.Label(settings_frame, text="Segmentation method:").grid(row=2, column=0, sticky="w", pady=5)
        self.method_box = ttk.Combobox(settings_frame, values=method_list, state="readonly")
        self.method_box.set("Otsu")
        self.method_box.grid(row=2, column=1, pady=5)
        
        #number of organs selector
        tk.Label(settings_frame, text="Number of Organs:").grid(row=3, column=0, sticky="w", pady=5)
        self.organ_no_box = ttk.Combobox(settings_frame, values=self.organ_numbers, state="readonly")
        self.organ_no_box.set(1)
        self.organ_no_box.grid(row=3, column=1, pady=5)
        
        tk.Button(settings_frame,text="Start Segmentation",command=self.start_segmentation).grid(row=4,column=1,sticky="w", pady=5)

        # ---- Status Frame ----
        status_frame = tk.Frame(self)
        status_frame.grid(row=2)

        self.message_var = tk.Message(status_frame, width=300)
        self.message_var.pack()

    def modify_status(self,text):
        self.message_var.configure(text=text)

    def browse_for_image(self):
         image_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])    
         if os.path.exists(image_path):
             self.input_entry.delete(0,"end")
             self.input_entry.insert(0,image_path)
    
    def browse_for_folder(self):
        folder_path = filedialog.askdirectory(title="Open Output Directory")
        if os.path.exists(folder_path):
            self.output_entry.delete(0,"end")
            self.output_entry.insert(0,folder_path)    
    
    def start_segmentation(self):
        
        self.modify_status("Starting segmentation...")
        
        image_path = self.input_entry.get()
        folder_path = self.output_entry.get()
        method = self.method_box.get()
        
        if not os.path.exists(image_path):
            self.modify_status("Invalid input image path")
            return
        
        organ_segmentation_pipeline(image_path,folder_path,method,self)
        

if __name__ == "__main__":
    root = tk.Tk()
    AppGUI(root)
    root.mainloop()
"""
GUV Detector - Graphical User Interface
Simple GUI for automatic liposomes detection
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import ultralytics
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.stats import lognorm
import numpy as np
import sys

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class GUVDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUV Detector - Liposomes Detection")
        self.root.geometry("1000x800")
        
        # Variables
        self.folder_path = tk.StringVar()
        self.model_size = tk.StringVar(value="n")
        self.modality = tk.StringVar(value="rgb")
        self.split_factor = tk.IntVar(value=2)
        self.mu_per_pixel = tk.DoubleVar(value=0.3339)
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="GUV Detector", 
                                font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Folder selection
        ttk.Label(main_frame, text="Image Folder:", 
                  font=('Helvetica', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.folder_path, 
                  width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse...", 
                   command=self.browse_folder).grid(row=1, column=2, pady=5)
        
        # Model size
        ttk.Label(main_frame, text="Model Size:", 
                  font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_size, 
                                   values=["n", "s", "m", "l", "x"], state="readonly", width=20)
        model_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Modality
        ttk.Label(main_frame, text="Image Modality:", 
                  font=('Helvetica', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=5)
        modality_combo = ttk.Combobox(main_frame, textvariable=self.modality, 
                                      values=["rgb", "grey"], state="readonly", width=20)
        modality_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Split factor
        ttk.Label(main_frame, text="Split Factor:", 
                  font=('Helvetica', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=5)
        split_frame = ttk.Frame(main_frame)
        split_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(split_frame, text="2 (4 sub-images)", 
                        variable=self.split_factor, value=2).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(split_frame, text="4 (16 sub-images)", 
                        variable=self.split_factor, value=4).pack(side=tk.LEFT, padx=5)
        
        # Conversion factor
        ttk.Label(main_frame, text="μm per pixel:", 
                  font=('Helvetica', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.mu_per_pixel, 
                  width=20).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Run button
        self.run_button = ttk.Button(main_frame, text="Run Detection", 
                                     command=self.run_detection, 
                                     style='Accent.TButton')
        self.run_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                      font=('Helvetica', 10))
        self.status_label.grid(row=8, column=0, columnspan=3, pady=5)
        
        # Notebook for tabs (Results and Plot)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=9, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Text Results
        result_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(result_frame, text="Text Results")
        
        self.result_text = tk.Text(result_frame, height=15, width=70, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # Tab 2: Plot
        self.plot_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.plot_frame, text="Size Distribution Plot")
        
        # Placeholder label for plot
        self.plot_placeholder = ttk.Label(self.plot_frame, 
                                         text="Run detection to see the plot", 
                                         font=('Helvetica', 12))
        self.plot_placeholder.pack(expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_path.set(folder)
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def append_result(self, message):
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_detection(self):
        # Validate inputs
        if not self.folder_path.get():
            messagebox.showerror("Error", "Please select an image folder!")
            return
        
        if not os.path.exists(self.folder_path.get()):
            messagebox.showerror("Error", "Selected folder does not exist!")
            return
        
        # Disable button and start progress
        self.run_button.config(state='disabled')
        self.progress.start()
        self.result_text.delete(1.0, tk.END)
        
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.plot_placeholder = ttk.Label(self.plot_frame, 
                                         text="Processing...", 
                                         font=('Helvetica', 12))
        self.plot_placeholder.pack(expand=True)
        
        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.perform_detection)
        thread.start()
    
    def perform_detection(self):
        try:
            self.update_status("Loading model...")
            model_relative = os.path.join('model', self.modality.get(), 
                              f'yolo11_{self.model_size.get()}', 'best.pt')
            model_path = resource_path(model_relative)
            
            if not os.path.exists(model_path):
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Model not found at: {model_path}"))
                return
            
            model = ultralytics.YOLO(model_path)
            
            self.update_status("Processing input folder...")
            sub_folder = self.processing_input_folder(
                self.folder_path.get(), self.split_factor.get())
            
            self.update_status("Running predictions...")
            results = model.predict(source=sub_folder, save=True, save_txt=True, 
                                   save_conf=True, project=sub_folder)
            
            if not os.path.exists(os.path.join(sub_folder, 'predict')):
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "No predictions found. Please check the model and input folder."))
                return
            
            self.update_status("Analyzing results...")
            self.analyze_results(sub_folder)
            
            self.update_status("Complete! ✓")
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", f"Detection complete!\nResults saved in:\n{sub_folder}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.update_status(f"Error: {str(e)}")
        
        finally:
            self.progress.stop()
            self.run_button.config(state='normal')
    
    def processing_input_folder(self, folder, split_factor):
        if split_factor not in [2, 4]:
            raise ValueError("split_factor must be 2 or 4.")
        
        files = os.listdir(folder)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            raise ValueError("No images found in folder!")
        
        first_image_path = os.path.join(folder, image_files[0])
        with Image.open(first_image_path) as img:
            reference_size = img.size
        
        processing_images = [first_image_path]
        for img_name in image_files[1:]:
            img_path = os.path.join(folder, img_name)
            with Image.open(img_path) as img:
                if img.size == reference_size:
                    processing_images.append(img_path)
        
        sub_folder = os.path.join(folder, 'processing_images')
        os.makedirs(sub_folder, exist_ok=True)
        
        self.append_result(f"Processing {len(processing_images)} images...")
        
        for img_path in processing_images:
            img = Image.open(img_path)
            w, h = img.size
            step_w = w // split_factor
            step_h = h // split_factor
            
            crop_boxes = []
            for i in range(split_factor):
                for j in range(split_factor):
                    left = j * step_w
                    upper = i * step_h
                    right = left + step_w
                    lower = upper + step_h
                    crop_boxes.append((left, upper, right, lower))
            
            for i, box in enumerate(crop_boxes):
                cropped_img = img.crop(box)
                cropped_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_crop{i+1}.png"
                cropped_path = os.path.join(sub_folder, cropped_name)
                cropped_img.save(cropped_path)
        
        self.append_result("Input folder processing complete!")
        return sub_folder
    
    def read_pred_boxes(self, pred_path, conf_thresh):
        boxes = []
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        cls, xc, yc, w, h, conf = map(float, parts[:6])
                        if conf >= conf_thresh:
                            boxes.append([xc, yc, w, h, conf])
        return boxes
    
    def display_plot(self, fig):
        """Display matplotlib figure in the GUI"""
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Create canvas for the plot
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Switch to plot tab
        self.notebook.select(1)
    
    def analyze_results(self, sub_folder):
        prediction_folder = os.path.join(sub_folder, 'predict', 'labels')
        
        count_edge = 0
        count_inter = 0
        conf_thresh = 0.25
        dim_list = []
        
        for i in os.listdir(prediction_folder):
            if i.endswith('.txt'):
                pred_path = os.path.join(prediction_folder, i)
                image_name = os.path.join(sub_folder, 'predict', i.replace('.txt', '.jpg'))
                w, h = Image.open(image_name).size
                boxes = self.read_pred_boxes(pred_path, conf_thresh)
                
                for bbox in boxes:
                    w_guv = bbox[2] * w 
                    h_guv = bbox[3] * h
                    max_dim = max(w_guv, h_guv)
                    min_dim = min(w_guv, h_guv)
                    
                    if min_dim <= 0.5 * max_dim:
                        count_edge += 1
                    else:
                        count_inter += 1
                    
                    dim = max_dim * self.mu_per_pixel.get()
                    dim_list.append(dim)
        
        if not dim_list:
            self.append_result("No GUVs detected!")
            return
        
        # Generate plot
        fig = Figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        
        bin_width = 5
        bins = np.arange(0, max(dim_list) + bin_width, bin_width)
        ax.hist(dim_list, bins=bins, color='chocolate', alpha=0.5, density=True)
        
        median = np.median(dim_list)
        first_quartile = np.percentile(dim_list, 25)
        third_quartile = np.percentile(dim_list, 75)
        
        ax.axvline(median, color='darkred', linestyle='dashed', linewidth=3, 
                   label=f'Median: {median:.2f} μm')
        ax.axvline(first_quartile, color='red', linestyle='dashed', linewidth=3, 
                   label=f'Q1: {first_quartile:.2f} μm')
        ax.axvline(third_quartile, color='red', linestyle='dashed', linewidth=3, 
                   label=f'Q3: {third_quartile:.2f} μm')
        
        shape, loc, scale = lognorm.fit(dim_list, floc=0)
        x = np.linspace(0, max(dim_list), 1000)
        n_guv = len(dim_list)
        mu = np.log(scale)
        sigma = shape
        pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
        
        ax.plot(x, pdf, 'k-', linewidth=3, label=(
            f'Log-normal fit\nμ={mu:.2f}, σ={sigma:.2f}\n'
            f'Total GUVs: {n_guv}'
        ))
        
        ax.set_xlabel('GUV Diameter (μm)', fontsize=20)
        ax.set_ylabel('Density of GUVs', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        ax.grid(linestyle=':')
        
        # Save plot to file
        plot_path = os.path.join(sub_folder, 'GUV_size_distribution.pdf')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Display plot in GUI
        self.root.after(0, lambda: self.display_plot(fig))
        
        # Display results
        self.append_result(f"\n=== RESULTS ===")
        self.append_result(f"Total GUVs detected: {n_guv}")
        self.append_result(f"Edge GUVs: {count_edge}")
        self.append_result(f"Interior GUVs: {count_inter}")
        self.append_result(f"\nSize statistics:")
        self.append_result(f"  Median: {median:.2f} μm")
        self.append_result(f"  Q1: {first_quartile:.2f} μm")
        self.append_result(f"  Q3: {third_quartile:.2f} μm")
        self.append_result(f"\nLog-normal parameters:")
        self.append_result(f"  μ: {mu:.2f}")
        self.append_result(f"  σ: {sigma:.2f}")
        self.append_result(f"\nPlot saved: {plot_path}")
        self.append_result(f"\n=== List of GUV dimensions (μm) ===")
        # Format the list nicely with 5 values per line
        for i in range(0, len(dim_list), 5):
            chunk = dim_list[i:i+5]
            formatted_chunk = ", ".join([f"{d:.2f}" for d in chunk])
            self.append_result(f"  {formatted_chunk}")


def main():
    root = tk.Tk()
    app = GUVDetectorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
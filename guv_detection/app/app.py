"""
GUV Detector - Graphical User Interface (cleaned)
Interactive per-image calibration; processing starts only after calibration.
"""
import os
import math
import threading
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import ultralytics
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.stats import lognorm
import numpy as np

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class CalibrationWindow:
    """Modal window: click 2 points, enter real distance (μm) -> returns mu_per_pixel"""
    def __init__(self, parent, image_path, image_name, default_mu):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Calibrate: {image_name}")
        self.top.geometry("950x900")
        self.top.grab_set()  # modal
        self.parent = parent

        self.image_path = image_path
        self.image_name = image_name
        self.points = []
        self.pixel_distance = 0.0
        self.mu_per_pixel = None
        self.calibration_done = False
        self.default_mu = default_mu

        # load image and scale if large
        self.original_image = Image.open(image_path)
        self.display_image = self.original_image.copy()
        max_display = 800
        if max(self.display_image.size) > max_display:
            ratio = max_display / max(self.display_image.size)
            new_size = (int(self.display_image.size[0] * ratio), int(self.display_image.size[1] * ratio))
            self.display_image = self.display_image.resize(new_size, Image.LANCZOS)
            self.scale_factor = ratio
        else:
            self.scale_factor = 1.0

        self._build_ui()

    def _build_ui(self):
        ttk.Label(self.top, text="Click TWO points on the image to define a known distance",
                  font=('Helvetica', 12, 'bold')).pack(pady=6)
        ttk.Label(self.top, text="Then enter the REAL distance in μm and press Calculate",
                  font=('Helvetica', 10)).pack()

        canvas_frame = ttk.Frame(self.top)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = tk.Canvas(canvas_frame, width=self.display_image.width, height=self.display_image.height, cursor='crosshair')
        self.canvas.pack()
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.bind("<Button-1>", self.on_click)

        self.info_label = ttk.Label(self.top, text="Click FIRST point...", font=('Helvetica', 11), foreground='blue')
        self.info_label.pack(pady=4)

        entry_frame = ttk.Frame(self.top)
        entry_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(entry_frame, text="Real distance (μm):", font=('Helvetica', 11)).pack(side=tk.LEFT)
        self.distance_entry = ttk.Entry(entry_frame, width=12, font=('Helvetica', 11))
        self.distance_entry.pack(side=tk.LEFT, padx=6)
        self.distance_entry.config(state='disabled')

        btn_frame = ttk.Frame(self.top)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btn_frame, text="Reset Points", command=self.reset_points).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text=f"Skip (use default {self.default_mu.get()})", command=self.skip_calibration).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Calculate", command=self.calculate_calibration).pack(side=tk.LEFT, padx=4)

        self.done_button = ttk.Button(btn_frame, text="✓ DONE", command=self.confirm_calibration)
        self.done_button.pack(side=tk.RIGHT)
        self.done_button.config(state='disabled')

    def on_click(self, event):
        if len(self.points) >= 2:
            messagebox.showinfo("Info", "Two points already selected. Reset to pick new points.")
            return
        x, y = event.x, event.y
        self.points.append((x, y))
        r = 4
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='red', outline='yellow', width=2, tags='cal')
        self.canvas.create_text(x, y-14, text=f"P{len(self.points)}", fill='yellow', font=('Helvetica', 10, 'bold'), tags='cal')
        if len(self.points) == 1:
            self.info_label.config(text="Click SECOND point...", foreground='blue')
        else:
            # draw line and compute pixel distance (corrected for scale_factor)
            x0, y0 = self.points[0]
            x1, y1 = self.points[1]
            self.canvas.create_line(x0, y0, x1, y1, fill='yellow', width=3, tags='cal')
            dx = (x1 - x0) / self.scale_factor
            dy = (y1 - y0) / self.scale_factor
            self.pixel_distance = math.hypot(dx, dy)
            self.info_label.config(text=f"Pixel distance: {self.pixel_distance:.1f} px. Enter real distance and press Calculate.", foreground='green')
            self.distance_entry.config(state='normal')
            self.distance_entry.focus()

    def calculate_calibration(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Select two points first.")
            return
        try:
            real_distance = float(self.distance_entry.get())
            if real_distance <= 0:
                raise ValueError("Distance must be > 0")
            self.mu_per_pixel = real_distance / self.pixel_distance
            self.info_label.config(text=f"Calibration: {self.mu_per_pixel:.4f} μm/pixel", foreground='darkgreen')
            self.done_button.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error", f"Invalid distance: {e}")

    def reset_points(self):
        self.points = []
        self.pixel_distance = 0.0
        self.mu_per_pixel = None
        self.canvas.delete('cal')
        self.info_label.config(text="Click FIRST point...", foreground='blue')
        self.distance_entry.delete(0, tk.END)
        self.distance_entry.config(state='disabled')
        self.done_button.config(state='disabled')

    def skip_calibration(self):
        if messagebox.askyesno("Skip calibration", f"Skip calibration for {self.image_name}?\nDefault will be used."):
            self.mu_per_pixel = self.default_mu.get()
            self.done_button.config(state='normal')  # allow confirm
            # immediate destroy after a short delay to ensure UI update
            self.calibration_done = True
            self.top.destroy()

    def confirm_calibration(self):
        if self.mu_per_pixel is None:
            messagebox.showwarning("Warning", "Calculate or skip before confirming.")
            return
        self.calibration_done = True
        self.top.destroy()

    def get_calibration(self):
        # blocks until window destroyed
        self.top.wait_window()
        return self.mu_per_pixel if self.calibration_done else None


class GUVDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUV Detector with YOLOv11")
        self.root.geometry("1000x800")

        # Variables
        self.folder_path = tk.StringVar()
        self.model_size = tk.StringVar(value="n")
        self.modality = tk.StringVar(value="grey")
        self.split_factor = tk.IntVar(value=2)
        self.mu_per_pixel = tk.DoubleVar(value=0.3339)
        self.conf_thresh = tk.DoubleVar(value=0.25)
        self.use_interactive_calibration = tk.BooleanVar(value=True)

        # calibration store: keys are original filenames (with extension)
        self.calibration_data = {}

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        ttk.Label(main, text="GUV Detector", font=('Helvetica', 20, 'bold')).grid(row=0, column=0, columnspan=3, pady=6)

        ttk.Label(main, text="Image Folder:", font=('Helvetica', 12)).grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(main, textvariable=self.folder_path, width=50).grid(row=1, column=1, padx=4, sticky=tk.W)
        ttk.Button(main, text="Browse...", command=self.browse_folder).grid(row=1, column=2, padx=4)

        ttk.Label(main, text="Model Size:", font=('Helvetica', 12)).grid(row=2, column=0, sticky=tk.W)
        ttk.Combobox(main, textvariable=self.model_size, values=["n","s","m","l","x"], state='readonly', width=18).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(main, text="Modality:", font=('Helvetica', 12)).grid(row=3, column=0, sticky=tk.W)
        ttk.Combobox(main, textvariable=self.modality, values=["rgb","grey"], state='readonly', width=18).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(main, text="Split factor:", font=('Helvetica', 12)).grid(row=4, column=0, sticky=tk.W)
        sf_frame = ttk.Frame(main)
        sf_frame.grid(row=4, column=1, sticky=tk.W)
        ttk.Radiobutton(sf_frame, text="2 (4)", variable=self.split_factor, value=2).pack(side=tk.LEFT)
        ttk.Radiobutton(sf_frame, text="4 (16)", variable=self.split_factor, value=4).pack(side=tk.LEFT)

        ttk.Label(main, text="Conf threshold:", font=('Helvetica', 12)).grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(main, textvariable=self.conf_thresh, width=10).grid(row=5, column=1, sticky=tk.W)

        ttk.Label(main, text="μm", font=('Helvetica', 12)).grid(row=6, column=0, sticky=tk.W)
        ttk.Entry(main, textvariable=self.mu_per_pixel, width=10).grid(row=6, column=1, sticky=tk.W)

        ttk.Checkbutton(main, text="Interactive calibration per image", variable=self.use_interactive_calibration).grid(row=7, column=1, sticky=tk.W, pady=4)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=8, column=0, columnspan=3, pady=8)
        self.run_button = ttk.Button(btn_frame, text="Run Detection", command=self.run_detection)
        self.run_button.pack(side=tk.LEFT, padx=6)
        self.reset_button = ttk.Button(btn_frame, text="Reset", command=self.reset_all, state='disabled')
        self.reset_button.pack(side=tk.LEFT, padx=6)

        self.progress = ttk.Progressbar(main, mode='determinate', length=420)
        self.progress.grid(row=9, column=0, columnspan=3, pady=6)

        self.status_label = ttk.Label(main, text="Ready", font=('Helvetica', 12))
        self.status_label.grid(row=10, column=0, columnspan=3, pady=4)

        # Notebook
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=11, column=0, columnspan=3, sticky=(tk.N,tk.S,tk.E,tk.W), pady=6)
        result_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(result_frame, text="Text Results")
        self.result_text = tk.Text(result_frame, height=18, wrap=tk.WORD)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.plot_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.plot_frame, text="Size Distribution Plot")
        self.plot_placeholder = ttk.Label(self.plot_frame, text="Run detection to see the plot", font=('Helvetica', 14))
        self.plot_placeholder.pack(expand=True)

        # make grid expand
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(11, weight=1)

    def browse_folder(self):
        f = filedialog.askdirectory(title="Select Image Folder")
        if f:
            self.folder_path.set(f)

    def append_result(self, txt):
        self.result_text.insert(tk.END, txt + "\n")
        self.result_text.see(tk.END)
        self.result_text.update_idletasks()

    def set_status(self, txt):
        self.status_label.config(text=txt)
        self.status_label.update_idletasks()

    def set_progress(self, value, maximum=100):
        self.progress['maximum'] = maximum
        self.progress['value'] = value
        self.progress.update_idletasks()

    def reset_all(self):
        self.result_text.delete(1.0, tk.END)
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self.plot_placeholder = ttk.Label(self.plot_frame, text="Run detection to see the plot", font=('Helvetica', 14))
        self.plot_placeholder.pack(expand=True)
        self.calibration_data = {}
        self.set_status("Ready")
        self.set_progress(0, 100)
        self.run_button.config(state='normal')
        self.reset_button.config(state='disabled')

    def run_detection(self):
        folder = self.folder_path.get()
        if not folder:
            messagebox.showerror("Error", "Select image folder first.")
            return
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Folder not found.")
            return

        # disable buttons
        self.run_button.config(state='disabled')
        self.reset_button.config(state='disabled')
        self.result_text.delete(1.0, tk.END)
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self.plot_placeholder = ttk.Label(self.plot_frame, text="Processing...", font=('Helvetica', 14))
        self.plot_placeholder.pack(expand=True)

        # start worker thread
        t = threading.Thread(target=self._worker_run, daemon=True)
        t.start()

    def _worker_run(self):
        try:
            self.set_status("Collecting images...")
            folder = self.folder_path.get()
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not files:
                raise RuntimeError("No image files found in folder.")

            # ensure all same size (filter others)
            first_path = os.path.join(folder, files[0])
            with Image.open(first_path) as im:
                ref_size = im.size
            valid_files = []
            rejected = []
            for f in files:
                try:
                    with Image.open(os.path.join(folder, f)) as im:
                        if im.size == ref_size:
                            valid_files.append(f)
                        else:
                            rejected.append(f)
                except:
                    rejected.append(f)

            if not valid_files:
                raise RuntimeError("No valid images with consistent size found.")

            self.append_result(f"Found {len(valid_files)} valid images. Rejected {len(rejected)} files.")
            if rejected:
                for r in rejected:
                    self.append_result(f"  - ignored: {r}")

            # calibration (synchronous, modal windows)
            if self.use_interactive_calibration.get():
                self.append_result("\n=== CALIBRATION PHASE ===")
                for idx, img_name in enumerate(valid_files, start=1):
                    self.set_status(f"Calibrating image {idx}/{len(valid_files)}: {img_name}")
                    img_path = os.path.join(folder, img_name)
                    calib = CalibrationWindow(self.root, img_path, img_name, self.mu_per_pixel).get_calibration()
                    if calib is None:
                        # user cancelled -> abort
                        raise RuntimeError("Calibration cancelled by user.")
                    self.calibration_data[img_name] = calib
                    self.append_result(f"  {img_name}: {calib:.4f} μm/pixel")
            else:
                default_mu = self.mu_per_pixel.get()
                for img_name in valid_files:
                    self.calibration_data[img_name] = default_mu
                self.append_result(f"Using default calibration for {len(valid_files)} images: {default_mu} μm/pixel")

            # splitting images
            self.set_status("Splitting images...")
            self.append_result("\n=== SPLITTING IMAGES ===")
            sub_folder = self._create_unique_folder(folder)

            total = len(valid_files)
            self.set_progress(0, total)
            for idx, img_name in enumerate(valid_files, start=1):
                img_path = os.path.join(folder, img_name)
                img = Image.open(img_path)
                w, h = img.size
                sf = int(self.split_factor.get())
                step_w = w // sf
                step_h = h // sf
                crop_idx = 0
                for i in range(sf):
                    for j in range(sf):
                        left = j * step_w
                        upper = i * step_h
                        right = left + step_w
                        lower = upper + step_h
                        crop = img.crop((left, upper, right, lower))
                        if self.modality.get() == 'grey':
                            crop = crop.convert('L')
                        crop_name = f"{os.path.splitext(img_name)[0]}_crop{crop_idx+1}.png"
                        crop_path = os.path.join(sub_folder, crop_name)
                        crop.save(crop_path)
                        crop_idx += 1
                self.append_result(f"Split {idx}/{total}: {img_name} -> {crop_idx} crops")
                self.set_progress(idx, total)
            self.set_progress(0, 100)
            self.set_status("Splitting done.")

            # load model
            self.set_status("Loading model...")
            model_rel = os.path.join('model', self.modality.get(), f'yolo11_{self.model_size.get()}', 'best.pt')
            model_path = resource_path(model_rel)
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model not found at: {model_path}")
            model = ultralytics.YOLO(model_path)

            # run predictions (save predictions in project=sub_folder)
            self.set_status("Running predictions (YOLO)...")
            # count images to predict for progress
            img_list = [f for f in os.listdir(sub_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            n_imgs = len(img_list)
            if n_imgs == 0:
                raise RuntimeError("No cropped images to predict.")
            self.set_progress(0, n_imgs)

            # use stream=True to iterate results and update progress
            results = model.predict(source=sub_folder, save=True, save_txt=True, save_conf=True,
                                    project=sub_folder, conf=float(self.conf_thresh.get()), stream=True)
            for idx, out in enumerate(results, start=1):
                # out is a Results object; we don't need to parse it here — we just update progress
                self.set_progress(idx, n_imgs)
                self.set_status(f"Predicting {idx}/{n_imgs}")
            self.set_status("Predictions done.")
            predict_root = os.path.join(sub_folder, 'predict')
            if not os.path.exists(predict_root):
                raise RuntimeError("YOLO did not produce a 'predict' folder.")

            # analysis
            self.set_status("Analyzing predictions...")
            self.append_result("\n=== ANALYSIS ===")
            labels_dir = os.path.join(predict_root, 'labels')
            if not os.path.exists(labels_dir):
                raise RuntimeError("No labels folder found in predictions.")

            label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            total_labels = len(label_files)
            self.set_progress(0, total_labels if total_labels>0 else 1)

            dim_list = []
            count_edge = 0
            count_inter = 0

            for idx, lab in enumerate(label_files, start=1):
                lab_path = os.path.join(labels_dir, lab)
                # try to find image corresponding to this label (png/jpg)
                base = os.path.splitext(lab)[0]
                img_candidate = None
                for ext in ('.png', '.jpg', '.jpeg'):
                    p = os.path.join(predict_root, base + ext)
                    if os.path.exists(p):
                        img_candidate = p
                        break
                if img_candidate is None:
                    # skip if image not found
                    self.append_result(f"  [warn] no image found for label {lab}")
                    self.set_progress(idx, total_labels)
                    continue

                # compute calibration for this cropped image (based on parent original name)
                cropped_name = os.path.basename(img_candidate)  # e.g. image001_crop1.png
                mu = self._get_mu_for_cropped(cropped_name)
                
                # read label boxes
                boxes = []
                with open(lab_path) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            try:
                                cls, xc, yc, wbox, hbox, conf = map(float, parts[:6])
                                if conf >= float(self.conf_thresh.get()):
                                    boxes.append((wbox, hbox))
                            except:
                                continue

                if not boxes:
                    self.append_result(f"  [info] no GUV in {cropped_name}")
                    self.set_progress(idx, total_labels)
                    continue

                # open image size
                w_img, h_img = Image.open(img_candidate).size
                for wbox, hbox in boxes:
                    W = wbox * w_img
                    H = hbox * h_img
                    max_dim = max(W, H)
                    min_dim = min(W, H)
                    if min_dim <= 0.5 * max_dim:
                        count_edge += 1
                    else:
                        count_inter += 1
                    dim_list.append(max_dim * mu)

                self.set_progress(idx, total_labels)
                self.set_status(f"Analyzing {idx}/{total_labels}")

            # final checks
            if not dim_list:
                self.append_result("No GUVs detected across all images.")
                self.set_status("Complete (no GUVs).")
                self.set_progress(100, 100)
                self.reset_button.config(state='normal')
                return

            # statistics and plot
            arr = np.array(dim_list)
            median = np.median(arr)
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            try:
                shape, loc, scale = lognorm.fit(arr, floc=0)
                mu_ln = float(np.log(scale))
                sigma_ln = float(shape)
            except Exception:
                shape = loc = scale = None
                mu_ln = sigma_ln = float('nan')

            # build plot
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

            # save and show
            plot_path = os.path.join(sub_folder, 'GUV_size_distribution.pdf')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.root.after(0, lambda: self._show_plot(fig))

            # textual results
            self.append_result("\n=== RESULTS ===")
            self.append_result(f"Total GUVs detected: {len(arr)}")
            self.append_result(f"Edge: {count_edge}")
            self.append_result(f"Interior: {count_inter}")
            self.append_result(f"Median: {median:.2f} μm  Q1: {q1:.2f} μm  Q3: {q3:.2f} μm")
            self.append_result(f"Log-normal μ={mu_ln:.2f}, σ={sigma_ln:.2f}")
            self.append_result(f"Plot saved: {plot_path}")
            self.append_result("\n=== Calibration data ===")
            for k,v in self.calibration_data.items():
                self.append_result(f"  {k}: {v:.4f} μm/pixel")

            self.append_result(f"\n=== List of GUV dimensions (μm) ===")
            for i in range(0, len(dim_list), 5):
                chunk = dim_list[i:i+5]
                formatted_chunk = ", ".join([f"{d:.2f}" for d in chunk])
                self.append_result(f"  {formatted_chunk}")

            # enable reset
            self.set_status("Complete ✓")
            self.set_progress(100, 100)
            self.reset_button.config(state='normal')

        except Exception as e:
            # show error in UI thread
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.set_status(f"Error: {e}")
            self.append_result(f"[ERROR] {e}")
            self.reset_button.config(state='normal')
        finally:
            self.run_button.config(state='disabled')

    def _get_mu_for_cropped(self, cropped_name):
        """
        Given cropped_name like 'image001_crop1.png' find original 'image001.ext' in calibration_data.
        """
        base = cropped_name.rsplit('_crop', 1)[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = base + ext
            if cand in self.calibration_data:
                return float(self.calibration_data[cand])
        # fallback default
        return self.mu_per_pixel.get()

    def _show_plot(self, fig):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.select(1)

    def _create_unique_folder(self, parent_folder):
        """
        Recursive creation of subfolder processing images
        """
        base_name = "processing_images"
        sub_folder = os.path.join(parent_folder, base_name)

        ## first implementation 
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
            return sub_folder

        # counter
        counter = 2
        while True:
            new_name = f"{base_name}_{counter}"
            sub_folder = os.path.join(parent_folder, new_name)
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
                return sub_folder
            counter += 1

def main():
    root = tk.Tk()
    app = GUVDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

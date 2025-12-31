import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FacePCAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Reconstruction with PCA")
        
        self.faces = fetch_olivetti_faces()
        self.X = self.faces.data
        self.images = self.faces.images
        self.n_samples, self.h, self.w = self.images.shape[0], self.images.shape[1], self.images.shape[2]
        
        self.pca = PCA(n_components=150, whiten=True)
        self.pca.fit(self.X)
        self.X_pca = self.pca.transform(self.X)
        self.mean_face = self.pca.mean_.reshape(self.h, self.w)
        
        self.current_components = np.zeros(150)
        
        self.setup_gui()
        self.update_image()
    
    def setup_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.img = self.ax.imshow(self.mean_face, cmap='gray')
        self.ax.axis('off')
        self.title = self.ax.set_title('Mean face + PCA components')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.sliders_frame = tk.Frame(self.root)
        self.sliders_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.sliders = []
        self.slider_vars = []
        
        for i in range(5):
            min_val = np.min(self.X_pca[:, i])
            max_val = np.max(self.X_pca[:, i])
            
            var = tk.DoubleVar(value=0)
            slider = ttk.Scale(
                self.sliders_frame,
                from_=min_val,
                to=max_val,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda val, idx=i: self.on_slider_change(idx, float(val)))
            
            label = tk.Label(self.sliders_frame, text=f'Component {i+1}:')
            
            label.grid(row=i, column=0, sticky=tk.W)
            slider.grid(row=i, column=1, sticky=tk.EW)
            
            self.sliders.append(slider)
            self.slider_vars.append(var)
        
        self.sliders_frame.columnconfigure(1, weight=1)
        
        self.btn_reset = tk.Button(
            self.sliders_frame,
            text="Reset",
            command=self.reset_sliders)
        self.btn_reset.grid(row=5, column=0, columnspan=2, pady=10)
    
    def on_slider_change(self, component_idx, value):
        self.current_components[component_idx] = value
        self.update_image()
    
    def update_image(self):
        reconstructed_face = self.pca.inverse_transform(self.current_components).reshape(self.h, self.w)
        self.img.set_array(reconstructed_face)
        
        components_text = ", ".join([f"{i+1}: {val:.1f}" for i, val in enumerate(self.current_components[:5])])
        self.title.set_text(f'PCA components: {components_text}')
        
        self.canvas.draw()
    
    def reset_sliders(self):
        for i in range(5):
            self.slider_vars[i].set(0)
            self.current_components[i] = 0
        self.update_image()
    
    def on_close(self):
        plt.close(self.fig)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacePCAApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
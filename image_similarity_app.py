import cv2
import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import time
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Similarity Contest Analyzer")
        self.root.geometry("900x650")
        self.root.minsize(900, 650)
        
        # Variables
        self.original_dir = tk.StringVar()
        self.participants_dir = tk.StringVar()
        self.status = tk.StringVar()
        self.progress = tk.DoubleVar()
        self.results_df = None
        self.current_comparison = None
        
        # Create GUI
        self.create_widgets()
        
        # Set initial status
        self.status.set("Ready to start. Please select folders.")
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Directory selection section
        dir_frame = ttk.LabelFrame(main_frame, text="Directory Selection", padding="10")
        dir_frame.pack(fill=tk.X, pady=10)
        
        # Original images directory
        ttk.Label(dir_frame, text="Original Images Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(dir_frame, textvariable=self.original_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_original_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # Participants directory
        ttk.Label(dir_frame, text="Participants Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(dir_frame, textvariable=self.participants_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_participants_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # Control section
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Run Analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        ttk.Label(progress_frame, textvariable=self.status).pack(anchor=tk.W, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabbed interface
        tab_control = ttk.Notebook(results_frame)
        
        # Tab 1: Table view
        self.table_tab = ttk.Frame(tab_control)
        tab_control.add(self.table_tab, text="Scores Table")
        
        # Scrollable table
        table_frame = ttk.Frame(self.table_tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars for the table
        table_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        table_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        
        # Treeview for the table
        self.score_table = ttk.Treeview(table_frame, 
                                        yscrollcommand=table_scroll_y.set, 
                                        xscrollcommand=table_scroll_x.set)
        
        table_scroll_y.config(command=self.score_table.yview)
        table_scroll_x.config(command=self.score_table.xview)
        
        table_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        table_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.score_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Tab 2: Comparison view
        self.compare_tab = ttk.Frame(tab_control)
        tab_control.add(self.compare_tab, text="Image Comparison")
        
        # Create comparison controls
        compare_controls = ttk.Frame(self.compare_tab)
        compare_controls.pack(fill=tk.X, pady=5)
        
        ttk.Label(compare_controls, text="Participant:").pack(side=tk.LEFT, padx=5)
        self.participant_combo = ttk.Combobox(compare_controls, width=15, state="readonly")
        self.participant_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(compare_controls, text="Image:").pack(side=tk.LEFT, padx=5)
        self.image_combo = ttk.Combobox(compare_controls, width=15, state="readonly")
        self.image_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(compare_controls, text="Compare", command=self.show_comparison).pack(side=tk.LEFT, padx=20)
        
        # Frame for comparison figure
        self.compare_frame = ttk.Frame(self.compare_tab)
        self.compare_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Finish setting up tabs
        tab_control.pack(fill=tk.BOTH, expand=True)
        
    def browse_original_dir(self):
        directory = filedialog.askdirectory(title="Select Original Images Folder")
        if directory:
            self.original_dir.set(directory)
            
    def browse_participants_dir(self):
        directory = filedialog.askdirectory(title="Select Participants Folder")
        if directory:
            self.participants_dir.set(directory)
    
    def start_analysis(self):
        # Validate directories
        if not os.path.isdir(self.original_dir.get()):
            self.status.set("Error: Original images directory is invalid!")
            return
            
        if not os.path.isdir(self.participants_dir.get()):
            self.status.set("Error: Participants directory is invalid!")
            return
        
        # Start analysis in a separate thread
        self.progress.set(0)
        threading.Thread(target=self.run_analysis, daemon=True).start()
    
    def run_analysis(self):
        try:
            # Get directories
            original_dir = self.original_dir.get()
            participants_dir = self.participants_dir.get()
            
            # Get list of original images
            original_images = [f for f in os.listdir(original_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not original_images:
                self.status.set("Error: No images found in original directory!")
                return
            
            # Get list of participants (folders inside participants_dir)
            participants = [d for d in os.listdir(participants_dir) 
                           if os.path.isdir(os.path.join(participants_dir, d))]
            
            if not participants:
                self.status.set("Error: No participant folders found!")
                return
            
            # Initialize results DataFrame
            columns = ['Participant', 'Image', 'Total_Score', 
                      'Structural_Similarity', 'Color_Similarity', 
                      'Edge_Similarity', 'Feature_Similarity']
            
            results = []
            
            # Calculate total comparisons for progress tracking
            total_comparisons = len(participants) * len(original_images)
            comparisons_done = 0
            
            # Process each participant
            for participant in participants:
                participant_dir = os.path.join(participants_dir, participant)
                
                # Process each original image
                for img_name in original_images:
                    self.status.set(f"Processing {participant}: {img_name}")
                    
                    original_img_path = os.path.join(original_dir, img_name)
                    participant_img_path = os.path.join(participant_dir, img_name)
                    
                    # Check if participant has the corresponding image
                    if os.path.exists(participant_img_path):
                        try:
                            # Calculate similarity
                            result = self.calculate_image_similarity(original_img_path, participant_img_path)
                            
                            # Add to results
                            results.append({
                                'Participant': participant,
                                'Image': img_name,
                                'Total_Score': result['total_score'],
                                'Structural_Similarity': result['component_scores']['structural_similarity'],
                                'Color_Similarity': result['component_scores']['color_similarity'],
                                'Edge_Similarity': result['component_scores']['edge_similarity'],
                                'Feature_Similarity': result['component_scores']['feature_similarity']
                            })
                        except Exception as e:
                            self.status.set(f"Error processing {participant}/{img_name}: {str(e)}")
                            # Add failed entry with 0 scores
                            results.append({
                                'Participant': participant,
                                'Image': img_name,
                                'Total_Score': 0,
                                'Structural_Similarity': 0,
                                'Color_Similarity': 0,
                                'Edge_Similarity': 0,
                                'Feature_Similarity': 0
                            })
                    else:
                        # Image missing, add entry with 0 scores
                        results.append({
                            'Participant': participant,
                            'Image': img_name,
                            'Total_Score': 0,
                            'Structural_Similarity': 0,
                            'Color_Similarity': 0,
                            'Edge_Similarity': 0,
                            'Feature_Similarity': 0
                        })
                    
                    # Update progress
                    comparisons_done += 1
                    progress_percent = (comparisons_done / total_comparisons) * 100
                    self.progress.set(progress_percent)
                    
            # Create DataFrame
            self.results_df = pd.DataFrame(results)
            
            # Update status
            self.status.set("Analysis complete!")
            
            # Update UI with results (in main thread)
            self.root.after(0, self.update_results_ui)
            
        except Exception as e:
            self.status.set(f"Error: {str(e)}")
    
    def update_results_ui(self):
        if self.results_df is None or self.results_df.empty:
            return
            
        # Clear existing table
        for item in self.score_table.get_children():
            self.score_table.delete(item)
            
        # Configure columns
        self.score_table['columns'] = list(self.results_df.columns)
        self.score_table['show'] = 'headings'
        
        for col in self.results_df.columns:
            self.score_table.heading(col, text=col)
            self.score_table.column(col, width=100, anchor=tk.CENTER)
        
        # Add data
        for _, row in self.results_df.iterrows():
            values = list(row)
            # Format float values
            for i, val in enumerate(values):
                if isinstance(val, float):
                    values[i] = f"{val:.2f}"
            self.score_table.insert('', tk.END, values=values)
            
        # Update comparison dropdowns
        participants = sorted(self.results_df['Participant'].unique())
        self.participant_combo['values'] = participants
        if participants:
            self.participant_combo.current(0)
            
        images = sorted(self.results_df['Image'].unique())
        self.image_combo['values'] = images
        if images:
            self.image_combo.current(0)
    
    def show_comparison(self):
        if self.results_df is None:
            return
            
        participant = self.participant_combo.get()
        image = self.image_combo.get()
        
        if not participant or not image:
            return
            
        # Get paths
        original_img_path = os.path.join(self.original_dir.get(), image)
        participant_img_path = os.path.join(self.participants_dir.get(), participant, image)
        
        # Check if both images exist
        if not os.path.exists(original_img_path):
            self.status.set(f"Error: Original image {image} not found!")
            return
            
        if not os.path.exists(participant_img_path):
            self.status.set(f"Error: Participant image {participant}/{image} not found!")
            return
            
        # Calculate similarity and get comparison
        try:
            result = self.calculate_image_similarity(original_img_path, participant_img_path)
            
            # Clear previous comparison
            for widget in self.compare_frame.winfo_children():
                widget.destroy()
                
            # Create figure
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(result['comparison_image'], cv2.COLOR_BGR2RGB))
            plt.title(f"Similarity: {result['total_score']:.2f}%")
            
            # Add component scores as text
            scores = result['component_scores']
            score_text = f"""
            Structural: {scores['structural_similarity']:.2f}%
            Color: {scores['color_similarity']:.2f}%
            Edge: {scores['edge_similarity']:.2f}%
            Feature: {scores['feature_similarity']:.2f}%
            """
            plt.figtext(0.15, 0.05, score_text, fontsize=10)
            plt.axis('off')
            
            # Display in UI
            canvas = FigureCanvasTkAgg(fig, master=self.compare_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.status.set(f"Error generating comparison: {str(e)}")
    
    def export_csv(self):
        if self.results_df is None or self.results_df.empty:
            self.status.set("No results to export!")
            return
            
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Results As"
        )
        
        if file_path:
            try:
                self.results_df.to_csv(file_path, index=False)
                self.status.set(f"Results exported to {file_path}")
            except Exception as e:
                self.status.set(f"Error exporting CSV: {str(e)}")
    
    def calculate_image_similarity(self, img1_path, img2_path):
        """
        Calculate similarity score between two images based on multiple criteria.
        Returns a score from 0-100 where 100 is perfect similarity.
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        # Check if images loaded successfully
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
        
        # Resize images to the same dimensions for comparison
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1_resized = cv2.resize(img1, (width, height))
        img2_resized = cv2.resize(img2, (width, height))
        
        # Convert to grayscale for some comparisons
        img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Structural Similarity (composition, layout, overall appearance)
        ssim_score, _ = ssim(img1_gray, img2_gray, full=True)
        
        # 2. Color histogram comparison
        # Convert to HSV for better color representation
        img1_hsv = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist1_h = cv2.calcHist([img1_hsv], [0], None, [180], [0, 180])
        hist1_s = cv2.calcHist([img1_hsv], [1], None, [256], [0, 256])
        hist1_v = cv2.calcHist([img1_hsv], [2], None, [256], [0, 256])
        
        hist2_h = cv2.calcHist([img2_hsv], [0], None, [180], [0, 180])
        hist2_s = cv2.calcHist([img2_hsv], [1], None, [256], [0, 256])
        hist2_v = cv2.calcHist([img2_hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1_h, hist1_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_s, hist1_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_v, hist1_v, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_h, hist2_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_s, hist2_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_v, hist2_v, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms
        h_score = cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CORREL)
        s_score = cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CORREL)
        v_score = cv2.compareHist(hist1_v, hist2_v, cv2.HISTCMP_CORREL)
        color_score = (h_score + s_score + v_score) / 3
        
        # 3. Edges (details and shapes)
        # Apply Canny edge detection
        edges1 = cv2.Canny(img1_gray, 100, 200)
        edges2 = cv2.Canny(img2_gray, 100, 200)
        
        # Compare edges using template matching
        edge_similarity = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # 4. Feature matching (detailed elements)
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        # Initialize feature matcher
        feature_score = 0
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            feature_score = len(good_matches) / max(len(kp1), len(kp2), 1)
        
        # Calculate the weighted average for final score
        weights = {
            'structural': 0.35,  # Composition and overall similarity
            'color': 0.25,       # Color palette matching
            'edges': 0.20,       # Details and shapes
            'features': 0.20     # Specific elements and content
        }
        
        final_score = (
            weights['structural'] * ssim_score + 
            weights['color'] * color_score + 
            weights['edges'] * edge_similarity + 
            weights['features'] * feature_score
        ) * 100
        
        # Ensure score is between 0 and 100
        final_score = max(0, min(100, final_score))
        
        # Visual output for comparison
        comparison = np.concatenate((img1_resized, img2_resized), axis=1)
        cv2.putText(comparison, f"Similarity: {final_score:.2f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return {
            'total_score': final_score,
            'component_scores': {
                'structural_similarity': ssim_score * 100,
                'color_similarity': color_score * 100,
                'edge_similarity': edge_similarity * 100,
                'feature_similarity': feature_score * 100
            },
            'comparison_image': comparison
        }

# Create executable with PyInstaller
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSimilarityApp(root)
    root.mainloop()
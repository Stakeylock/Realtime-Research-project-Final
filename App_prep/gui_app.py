import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from PIL import Image, ImageTk

# Add the parent directory to the Python path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference_app
import llm_utils
import db_utils

class BreastCancerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Breast Cancer Risk Prediction")
        self.geometry("1000x800")

        # --- Initialize Backend ---
        print("Initializing database...")
        db_utils.init_db()
        print("Database initialized.")

        print("Loading models...")
        if not inference_app.load_all_models():
            messagebox.showerror("Error", "Failed to load models. The application will now close.")
            self.destroy()
            return
        print("Models loaded successfully.")

        # --- Main UI Structure ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.main_tab = ttk.Frame(self.notebook)
        self.patients_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text='Run Prediction')
        self.notebook.add(self.patients_tab, text='Patient History')

        self.create_main_widgets()
        self.create_patients_widgets()

        self.patient_id = None

    def create_main_widgets(self):
        # Main frame with a canvas and scrollbar
        canvas = tk.Canvas(self.main_tab)
        scrollbar = ttk.Scrollbar(self.main_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Questionnaire Frame ---
        q_frame = ttk.LabelFrame(scrollable_frame, text="Patient Questionnaire", padding="10")
        q_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.q_entries = {}
        questions = [
            "Patient ID", "Age", "Menopausal Status (Pre/Post)", 
            "Tumor Size (mm)", "Tumor Grade (1-3)", "Lymph Nodes Positive",
            "Hormone Receptor Status (Positive/Negative)", "Family History (Yes/No)"
        ]

        for i, q_text in enumerate(questions):
            label = ttk.Label(q_frame, text=q_text)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(q_frame, width=50)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            self.q_entries[q_text] = entry

        submit_q_button = ttk.Button(q_frame, text="Submit Questionnaire", command=self.submit_questionnaire)
        submit_q_button.grid(row=len(questions), column=0, columnspan=2, pady=10)

        # --- File Upload Frame ---
        f_frame = ttk.LabelFrame(scrollable_frame, text="Upload Data", padding="10")
        f_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.image_path_label = ttk.Label(f_frame, text="No image selected.")
        self.image_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        upload_img_button = ttk.Button(f_frame, text="Upload Mammogram Image", command=self.upload_image)
        upload_img_button.grid(row=0, column=0, padx=5, pady=5)

        self.gene_path_label = ttk.Label(f_frame, text="No gene file selected.")
        self.gene_path_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        upload_gene_button = ttk.Button(f_frame, text="Upload Gene Data (.csv)", command=self.upload_gene_data)
        upload_gene_button.grid(row=1, column=0, padx=5, pady=5)

        # --- Run Prediction Frame ---
        run_frame = ttk.LabelFrame(scrollable_frame, text="Run Analysis", padding="10")
        run_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        run_button = ttk.Button(run_frame, text="Run Full Prediction", command=self.run_prediction)
        run_button.pack(pady=10)

        # --- Results Frame ---
        res_frame = ttk.LabelFrame(scrollable_frame, text="Results", padding="10")
        res_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.image_result_label = ttk.Label(res_frame)
        self.image_result_label.pack(pady=5)
        self.gene_result_label = ttk.Label(res_frame)
        self.gene_result_label.pack(pady=5)
        self.final_report_text = tk.Text(res_frame, height=15, width=100, wrap="word")
        self.final_report_text.pack(pady=5)

    def create_patients_widgets(self):
        # Frame for the list of patients
        list_frame = ttk.Frame(self.patients_tab)
        list_frame.pack(fill="x", padx=10, pady=5)

        self.patient_listbox = tk.Listbox(list_frame, height=10)
        self.patient_listbox.pack(side="left", fill="x", expand=True)
        self.patient_listbox.bind('<<ListboxSelect>>', self.on_patient_select)

        refresh_button = ttk.Button(list_frame, text="Refresh List", command=self.refresh_patient_list)
        refresh_button.pack(side="left", padx=10)

        # Frame for displaying patient details
        self.details_text = tk.Text(self.patients_tab, height=25, width=100, wrap="word")
        self.details_text.pack(pady=5, padx=10, fill="both", expand=True)

        self.refresh_patient_list()

    def submit_questionnaire(self):
        data = {key: entry.get() for key, entry in self.q_entries.items()}
        if not data["Patient ID"]:
            messagebox.showerror("Error", "Patient ID is required.")
            return
        
        self.patient_id = data["Patient ID"]
        db_utils.save_patient_data({'questionnaire': data})
        messagebox.showinfo("Success", f"Questionnaire for Patient ID {self.patient_id} saved.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Mammogram Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if self.image_path:
            self.image_path_label.config(text=os.path.basename(self.image_path))

    def upload_gene_data(self):
        self.gene_path = filedialog.askopenfilename(title="Select Gene Data File", filetypes=[("CSV Files", "*.csv")])
        if self.gene_path:
            self.gene_path_label.config(text=os.path.basename(self.gene_path))

    def run_prediction(self):
        if not self.patient_id:
            messagebox.showerror("Error", "Please submit the questionnaire first.")
            return

        # --- Image Prediction ---
        if hasattr(self, 'image_path') and self.image_path:
            try:
                original_img_pil, grid_img_pil, result_text, _, _ = inference_app.predict_image(self.image_path)
                db_utils.save_image_prediction(self.patient_id, self.image_path, result_text)
                
                # Display images
                img_tk = ImageTk.PhotoImage(grid_img_pil.resize((300, 300)))
                self.image_result_label.config(image=img_tk)
                self.image_result_label.image = img_tk # Keep a reference!

            except Exception as e:
                messagebox.showerror("Image Prediction Error", str(e))
        
        # --- Gene Prediction ---
        if hasattr(self, 'gene_path') and self.gene_path:
            try:
                result_text, _ = inference_app.predict_gene_expression(self.gene_path)
                db_utils.save_gene_prediction(self.patient_id, self.gene_path, result_text)
                self.gene_result_label.config(text=f"Gene Analysis Result: {result_text}")
            except Exception as e:
                messagebox.showerror("Gene Prediction Error", str(e))

        # --- Generate Final Report ---
        try:
            report_html = llm_utils.generate_report_with_groq(self.patient_id)
            db_utils.save_report(self.patient_id, report_html)
            self.final_report_text.delete('1.0', tk.END)
            self.final_report_text.insert(tk.END, "--- Final Report (Plain Text) ---\n")
            # A simple conversion from HTML for display, not perfect
            self.final_report_text.insert(tk.END, report_html.replace('<br>', '\n').replace('<h1>', '**').replace('</h1>', '**\n'))
            messagebox.showinfo("Success", "Full analysis complete and report generated.")
        except Exception as e:
            messagebox.showerror("Report Generation Error", str(e))

    def refresh_patient_list(self):
        self.patient_listbox.delete(0, tk.END)
        patients = db_utils.get_all_patients()
        for p in patients:
            self.patient_listbox.insert(tk.END, f"{p['id']} - {p['name']}")

    def on_patient_select(self, event):
        selected_indices = self.patient_listbox.curselection()
        if not selected_indices:
            return
        
        selected_item = self.patient_listbox.get(selected_indices[0])
        patient_id = selected_item.split(' - ')[0]
        
        records = db_utils.get_patient_records(patient_id)
        self.details_text.delete('1.0', tk.END)
        
        if not records:
            self.details_text.insert(tk.END, f"No records found for Patient ID: {patient_id}")
            return

        for record in records:
            self.details_text.insert(tk.END, f"--- Record: {record['timestamp']} ---\n")
            if record.get('questionnaire_data'):
                self.details_text.insert(tk.END, f"Questionnaire: {record['questionnaire_data']}\n")
            if record.get('image_predictions'):
                self.details_text.insert(tk.END, f"Image Predictions: {record['image_predictions']}\n")
            if record.get('gene_predictions'):
                self.details_text.insert(tk.END, f"Gene Predictions: {record['gene_predictions']}\n")
            if record.get('reports'):
                self.details_text.insert(tk.END, f"Reports: {len(record['reports'])} available.\n")
            self.details_text.insert(tk.END, "\n")

if __name__ == "__main__":
    app = BreastCancerApp()
    app.mainloop()
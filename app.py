import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import os


# Load the model and feature columns
model = joblib.load('knn_model_.joblib')
with open('feature_columns_.joblib', 'rb') as f:
    feature_columns = joblib.load(f)

# Create the main application window
root = tk.Tk()
root.title("Predict IIT Status")
root.geometry("600x600")
root.configure(bg="#f0f4f7")  # Light blue-gray background for a cleaner look

# Add a title label
title_label = tk.Label(root, text="IIT Status Prediction Tool", font=("Helvetica", 16, "bold"), bg="#f0f4f7")
title_label.pack(pady=20)

# Frame for form inputs
form_frame = tk.Frame(root, bg="#f0f4f7")
form_frame.pack(pady=10, padx=20, fill="x")

# Helper function to create labeled entries
def create_labeled_entry(frame, label_text):
    label = tk.Label(frame, text=label_text, font=("Helvetica", 12), bg="#f0f4f7")
    entry = tk.Entry(frame, font=("Helvetica", 12), width=30)
    label.pack(pady=5)
    entry.pack(pady=5)
    return entry

# Collect user input fields
entry_LGA = create_labeled_entry(form_frame, "LGA:")
entry_Sex = create_labeled_entry(form_frame, "Sex (M/F):")
entry_Adherence = create_labeled_entry(form_frame, "Adherence (Good/Bad):")
entry_ActiveStatus = create_labeled_entry(form_frame, "Active Status (Active/LTFU):")
entry_Viralsuppression = create_labeled_entry(form_frame, "Viral Suppression (Suppressed/Unsuppressed):")
entry_TransferIn = create_labeled_entry(form_frame, "Transfer In (Yes/No):")

# Function to predict IIT status for a single entry
def predict_single():
    data = {
        'LGA': [entry_LGA.get()],
        'Sex': [entry_Sex.get()],
        'Adherence': [entry_Adherence.get()],
        'ActiveStatus': [entry_ActiveStatus.get()],
        'Viralsuppression': [entry_Viralsuppression.get()],
        'TransferIn': [entry_TransferIn.get()]
    }
    input_data = pd.DataFrame(data)
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    # Prediction
    prediction = model.predict(input_data)[0]
    messagebox.showinfo("Prediction Result", f"The predicted IIT Status is: {prediction}")

# Predict button with styling
predict_button = tk.Button(root, text="Predict IIT Status", font=("Helvetica", 14), command=predict_single, bg="#4CAF50", fg="white")
predict_button.pack(pady=20, ipadx=10, ipady=5)

# Batch Prediction with file upload
def predict_batch():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        input_data = pd.read_csv(file_path)
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        predictions = model.predict(input_data)
        input_data['Predicted_IITStatus'] = predictions
        input_data.to_csv("batch_predictions.csv", index=False)
        messagebox.showinfo("Batch Prediction", "Batch predictions saved to 'batch_predictions.csv'.")

batch_button = tk.Button(root, text="Batch Prediction (Upload CSV)", font=("Helvetica", 14), command=predict_batch, bg="#2196F3", fg="white")
batch_button.pack(pady=20, ipadx=10, ipady=5)

# Run the application
root.mainloop()
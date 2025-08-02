import tkinter as tk
from tkinter import messagebox, scrolledtext
import requests
import json

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Input Data for /predict API")

        self.data = []

        tk.Label(root, text="Enter 10 numbers separated by commas:").pack(padx=10, pady=5)
        self.entry = tk.Entry(root, width=60)
        self.entry.pack(padx=10, pady=5)

        self.add_btn = tk.Button(root, text="Add Row", command=self.add_row)
        self.add_btn.pack(pady=5)

        tk.Label(root, text="Data added:").pack(padx=10, pady=5)
        self.text_area = scrolledtext.ScrolledText(root, width=70, height=10)
        self.text_area.pack(padx=10, pady=5)
        self.text_area.config(state='disabled')

        self.send_btn = tk.Button(root, text="Send to API", command=self.send_data)
        self.send_btn.pack(pady=10)

    def add_row(self):
        text = self.entry.get()
        try:
            nums = [float(x.strip()) for x in text.split(',')]
            if len(nums) != 10:
                messagebox.showerror("Error", "Please enter exactly 10 numbers.")
                return
            self.data.append(nums)
            self.entry.delete(0, tk.END)
            self.update_text_area()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers separated by commas.")

    def update_text_area(self):
        self.text_area.config(state='normal')
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, json.dumps({"data": self.data}, indent=2))
        self.text_area.config(state='disabled')

    def send_data(self):
        if not self.data:
            messagebox.showerror("Error", "No data to send.")
            return
        url = "http://127.0.0.1:8000/predict"
        headers = {"Content-Type": "application/json"}
        payload = {"data": self.data}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            messagebox.showinfo("Response", json.dumps(result, indent=2))
        except requests.exceptions.RequestException as e:
            messagebox.showerror("Request Failed", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictApp(root)
    root.mainloop()

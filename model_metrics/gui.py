import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NewsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BBC News Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#f5f5f5")
        
        # Check if model files exist
        required_files = ['bilstm_news_classifier.h5', 'tokenizer.pickle', 'config.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            messagebox.showerror("Error", f"Missing required model files: {', '.join(missing_files)}\n"
                                "Please ensure model is trained first.")
            root.destroy()
            return
        
        # Load model and artifacts
        try:
            print("Loading model and artifacts...")
            self.model = load_model('bilstm_news_classifier.h5')
            
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
                
            with open('config.pkl', 'rb') as f:
                config = pickle.load(f)
                
            self.max_sequence_length = config['max_sequence_length']
            self.category_mapping = config['category_mapping']
            self.index_to_category = {idx: category for category, idx in self.category_mapping.items()}
            
            print("Model and artifacts loaded successfully")
            print(f"Categories: {list(self.category_mapping.keys())}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            root.destroy()
            return
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title label
        title_label = tk.Label(self.root, 
                             text="BBC News Article Classifier",
                             font=("Arial", 18, "bold"),
                             bg="#f5f5f5")
        title_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(self.root,
                              text="Paste or type a news article below and click 'Classify'",
                              font=("Arial", 12),
                              bg="#f5f5f5")
        instructions.pack(pady=5)
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(self.root, 
                                                  width=80, 
                                                  height=15,
                                                  font=("Arial", 11),
                                                  wrap=tk.WORD)
        self.text_input.pack(pady=10, padx=20)
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg="#f5f5f5")
        button_frame.pack(pady=10)
        
        # Classify button
        self.classify_button = ttk.Button(button_frame, 
                                        text="Classify",
                                        command=self.classify_text,
                                        width=20)
        self.classify_button.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        self.clear_button = ttk.Button(button_frame, 
                                     text="Clear",
                                     command=self.clear_text,
                                     width=20)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        # Result frame
        result_frame = tk.LabelFrame(self.root, 
                                   text="Classification Result", 
                                   font=("Arial", 12),
                                   bg="#f5f5f5")
        result_frame.pack(pady=10, padx=20, fill="x")
        
        # Result labels
        self.result_category = tk.StringVar()
        self.result_confidence = tk.StringVar()
        
        # Category result
        category_label = tk.Label(result_frame, 
                                text="Category:",
                                font=("Arial", 12, "bold"),
                                bg="#f5f5f5")
        category_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.category_result = tk.Label(result_frame,
                                      textvariable=self.result_category,
                                      font=("Arial", 12),
                                      bg="#f5f5f5")
        self.category_result.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Confidence result
        confidence_label = tk.Label(result_frame,
                                  text="Confidence:",
                                  font=("Arial", 12, "bold"),
                                  bg="#f5f5f5")
        confidence_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.confidence_result = tk.Label(result_frame,
                                        textvariable=self.result_confidence,
                                        font=("Arial", 12),
                                        bg="#f5f5f5")
        self.confidence_result.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Sample articles section
        samples_frame = tk.LabelFrame(self.root,
                                    text="Sample Articles",
                                    font=("Arial", 12),
                                    bg="#f5f5f5")
        samples_frame.pack(pady=10, padx=20, fill="x")
        
        # Sample buttons
        sample_categories = ["business", "tech", "sport", "politics", "entertainment"]
        for i, category in enumerate(sample_categories):
            sample_btn = ttk.Button(samples_frame,
                                  text=category.capitalize(),
                                  command=lambda cat=category: self.load_sample(cat))
            sample_btn.grid(row=0, column=i, padx=5, pady=5)
            
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root,
                                 textvariable=self.status_var,
                                 bd=1,
                                 relief=tk.SUNKEN,
                                 anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = str(text).lower()
        text = ' '.join(text.split())
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text
    
    def classify_text(self):
        """Classify the input text"""
        try:
            # Get text from input area
            text = self.text_input.get("1.0", tk.END).strip()
            
            if not text:
                messagebox.showwarning("Warning", "Please enter some text to classify.")
                return
            
            self.status_var.set("Classifying...")
            self.root.update_idletasks()
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Convert to sequence
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
            
            # Make prediction
            prediction = self.model.predict(padded_sequence)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get category name
            category = self.index_to_category[predicted_class]
            
            # Show results
            self.result_category.set(category.capitalize())
            self.result_confidence.set(f"{confidence:.2%}")
            
            # Change color based on confidence
            color = self.get_confidence_color(confidence)
            self.confidence_result.config(fg=color)
            
            self.status_var.set(f"Classified as {category} with {confidence:.2%} confidence")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
    
    def get_confidence_color(self, confidence):
        """Return color based on confidence level"""
        if confidence >= 0.9:
            return "green"
        elif confidence >= 0.7:
            return "blue"
        elif confidence >= 0.5:
            return "orange"
        else:
            return "red"
    
    def clear_text(self):
        """Clear the input text area"""
        self.text_input.delete("1.0", tk.END)
        self.result_category.set("")
        self.result_confidence.set("")
        self.status_var.set("Ready")
    
    def load_sample(self, category):
        """Load a sample article for the selected category"""
        samples = {
            "business": """
            The economy is showing signs of recovery as businesses reopen and consumer spending increases. 
            Retail sales jumped 5.3% last month, indicating strong consumer confidence. 
            The stock market has also responded positively, with major indices reaching new highs. 
            Several companies have announced plans to expand operations and hire more workers in the coming months. 
            Economists are optimistic about growth prospects for the remainder of the year.
            """,
            
            "tech": """
            A new artificial intelligence system has been developed that can diagnose medical conditions with greater accuracy than human doctors. 
            The AI uses deep learning algorithms trained on millions of patient records to identify patterns and make predictions. 
            In trials, it achieved 95% accuracy in detecting early signs of disease. 
            Researchers say this technology could revolutionize healthcare, especially in regions with limited access to medical professionals. 
            Privacy concerns remain about how patient data will be handled and protected.
            """,
            
            "sport": """
            The national football team secured a dramatic victory in last night's championship final, winning 3-2 with a goal in the final minute of extra time. 
            The team's captain scored twice, including the winning goal, in what many are calling one of the greatest performances in the tournament's history. 
            Thousands of fans celebrated in the streets after the match, which marks the country's first major trophy in fifteen years. 
            The coach praised the team's determination and fighting spirit throughout the competition.
            """,
            
            "politics": """
            Parliament is set to vote on a controversial new bill next week that would reform the country's healthcare system. 
            The opposition has criticized the proposal, claiming it doesn't provide adequate funding for rural hospitals. 
            Meanwhile, the prime minister defended the bill during a press conference, arguing it would improve access to care for millions of citizens. 
            Political analysts suggest the vote will be close, with several key lawmakers still undecided. 
            Public opinion polls show the nation divided on the issue, with 48% in favor and 45% opposed.
            """,
            
            "entertainment": """
            The latest superhero movie broke box office records during its opening weekend, earning over $200 million domestically. 
            Critics have praised the film's special effects and performances, with many calling it the best in the franchise. 
            The lead actor revealed in an interview that preparation for the role involved six months of intensive training. 
            Fans are already speculating about potential sequels, with rumors that filming for the next installment will begin later this year. 
            The soundtrack has also topped music charts, featuring collaborations with several popular artists.
            """
        }
        
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", samples[category].strip())
        self.status_var.set(f"Loaded sample {category} article")

if __name__ == "__main__":
    root = tk.Tk()
    app = NewsClassifierApp(root)
    root.mainloop()
# Multilingual Customer Support Ticket Classifier

This project builds a machine learning model to classify customer support tickets based on their **subject** and **body** text. It supports multilingual data and predicts labels like ticket **priority**, **type**, or **queue** to help automate customer service workflows.

---

## Features

- Clean and preprocess multilingual ticket data  
- Use TF-IDF vectorization and Logistic Regression classifier  
- Evaluate model with accuracy and confusion matrix  
- Simple Flask web app for real-time predictions  
- Modular and extensible codebase  

---

## Setup & Installation

1. Clone this repo:

```bash
git clone https://github.com/yourusername/ticket-classifier.git](https://github.com/THANUAA/Multilingual-Customer-Support-Ticket-Classifier.git
cd ticket-classifier
```
2.Install dependencies:

pip install -r requirements.txt

3. Run the Flask app:
   
python app.py

4. Open your browser at http://localhost:5000

ticket_classifier/
├── data/                      # Raw dataset files
├── models/                    # Saved ML models & vectorizers
├── app/                       # Flask app files
├── notebook/                  # Jupyter Notebook with EDA & training
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

Usage
Enter ticket subject and body text in the web UI.

Click "Predict Priority" to get the predicted priority class.

Future Work
Use transformer-based embeddings (e.g. XLM-Roberta) for better accuracy.

Add multi-label classification (priority + type).

Deploy using Docker or cloud services.

License
MIT License

Author
Your Name — www.tharakaarawinda999@gmail.com
GitHub: https://github.com/THANUAA



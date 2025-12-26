📊 Sentiment Analysis using Classical ML & BERT

This project implements a sentiment analysis system for Google Play Store reviews using both traditional machine learning techniques and a fine-tuned BERT model. The goal is to compare classical NLP approaches with modern transformer-based models for multi-class sentiment classification.

🚀 Project Overview
   The system classifies user reviews into three sentiment classes:
        1. Negative
        2. Neutral
        3. Positive

    Two main approaches are used:
       1. TF-IDF + Classical Machine Learning
       2. Fine-tuned BERT (Transformer-based deep learning)
    The project demonstrates the full NLP pipeline, from data preprocessing and feature engineering to model training, evaluation, and inference.

    📂 Dataset
        Source: Google Play Store Reviews
        Columns used:
            content → Review text
            score → Rating (used to derive sentiment)
      Sentiment Mapping:
          Rating ≤ 2 → Negative
          Rating = 3 → Neutral
          Rating ≥ 4 → Positive


     🧠 Models & Techniques
       🔹 Classical Machine Learning
              Text Representation: TF-IDF Vectorization
          Models:
               1. Random Forest Classifier
               2. Decision Tree Classifier
               3. Gradient Boosting Classifier
          Evaluation:
                1. Precision, Recall, F1-Score
                2. Confusion Matrix
     🔹 Deep Learning (BERT)
            Model: bert-base-cased
            Frameworks:
                 1. PyTorch
                 2. Hugging Face Transformers
            Key Components:
                 1. Tokenization with attention masks
                 2. Custom PyTorch Dataset & DataLoader
                 3. Fine-tuning using Cross-Entropy Loss
                 4. AdamW optimizer with learning rate scheduler
             Training:
                 1. GPU acceleration (if available) 
                 2. Validation tracking
                 3. Best model checkpoint saving     

    🔄 Workflow
        1. Load and explore dataset
        2. Clean and preprocess text data
        3. Generate sentiment labels
        4. Train classical ML models using TF-IDF features
        5. Build and fine-tune BERT model
        6. Evaluate models on validation & test sets
        7. Perform sentiment prediction on raw text    


     📊 Results
         1. Classical ML models provide strong baselines with fast training.
         2. BERT significantly improves contextual understanding and overall accuracy.
         3. Performance is evaluated using:
             1. Accuracy
             2. Classification Report
             3. Confusion Matrix visualization    


    📁 Project Structure
        sentiment-analysis-bert/
        │
        ├── README.md
        ├── requirements.txt
        ├── sentiment_analysis.ipynb
        └── data/
        └── reviews.csv

     👤 Author

       Gasser Ahmed
         Data Science & Machine Learning Enthusiast
        📍 Focused on NLP, Deep Learning & Generative AI

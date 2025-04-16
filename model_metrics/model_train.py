import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import time

# Set random seed for reproducibility
np.random.seed(42)

def load_data(csv_path, sample_size=None):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Basic validation
    assert 'Text' in df.columns, "CSV must contain a 'Text' column"
    assert 'Category' in df.columns, "CSV must contain a 'Category' column"
    
    # Optional: Sample a subset of data for faster training
    if sample_size and sample_size < len(df):
        df = df.groupby('Category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))))
        )
        print(f"Sampled dataset shape: {df.shape}")
    
    # Check category distribution
    print("Category distribution:")
    print(df['Category'].value_counts())
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Category']
    )
    
    return train_df, val_df

def preprocess_text(text):
    # Basic preprocessing
    text = str(text).lower()
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Replace any characters that might cause issues
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

def prepare_text_data(train_df, val_df, max_words=20000, max_sequence_length=500):
    """
    Tokenize and prepare text data for BiLSTM model
    """
    # Preprocess text
    train_texts = [preprocess_text(text) for text in train_df['Text']]
    val_texts = [preprocess_text(text) for text in val_df['Text']]
    
    # Tokenize the text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)
    
    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    val_sequences = tokenizer.texts_to_sequences(val_texts)
    
    # Pad sequences to ensure uniform length
    X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)
    
    # Create category mapping
    unique_categories = np.union1d(train_df['Category'].unique(), val_df['Category'].unique())
    category_to_index = {category: idx for idx, category in enumerate(unique_categories)}
    index_to_category = {idx: category for category, idx in category_to_index.items()}
    
    # Convert categories to numerical indices
    y_train = train_df['Category'].map(category_to_index).values
    y_val = val_df['Category'].map(category_to_index).values
    
    # Get the number of classes
    num_classes = len(unique_categories)
    print(f"Number of classes: {num_classes}")
    print(f"Categories: {unique_categories}")
    print(f"Category mapping: {category_to_index}")
    
    return X_train, y_train, X_val, y_val, tokenizer, num_classes, category_to_index

def build_bilstm_model(vocab_size, embedding_dim=100, max_sequence_length=500, num_classes=5):
    """
    Build a BiLSTM model for text classification
    """
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=max_sequence_length))
    
    # BiLSTM layers
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer for multi-class classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model(model, X_val, y_val, category_mapping):
    """
    Evaluate the BiLSTM model
    """
    # Make predictions
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    
    print("\nDetailed Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Get class names for the report
    index_to_category = {idx: category for category, idx in category_mapping.items()}
    target_names = [index_to_category[i] for i in range(len(category_mapping))]
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(y_val))):
        true_category = index_to_category[y_val[i]]
        pred_category = index_to_category[y_pred[i]]
        print(f"  True: {true_category}, Predicted: {pred_category}")
    
    return accuracy, y_pred

def plot_training_history(history):
    """
    Plot training & validation loss and accuracy
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_model_artifacts(model, tokenizer, max_sequence_length, category_mapping):
    """
    Save the model and tokenizer
    """
    # Save the model
    model.save('bilstm_news_classifier.h5')
    
    # Save the tokenizer
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save max_sequence_length and category_mapping
    with open('config.pkl', 'wb') as f:
        pickle.dump({
            'max_sequence_length': max_sequence_length,
            'category_mapping': category_mapping
        }, f)
    
    print("Model artifacts saved:")
    print("  - bilstm_news_classifier.h5")
    print("  - tokenizer.pickle")
    print("  - config.pkl")

def main(csv_path, sample_size=None):
    # Load data
    train_df, val_df = load_data(csv_path, sample_size)
    
    # Set parameters
    max_words = 20000  # Maximum number of words in the vocabulary
    max_sequence_length = 500  # Maximum length of text sequences
    embedding_dim = 100  # Dimension of word embeddings
    
    # Prepare text data
    X_train, y_train, X_val, y_val, tokenizer, num_classes, category_mapping = prepare_text_data(
        train_df, val_df, max_words, max_sequence_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Build BiLSTM model
    model = build_bilstm_model(
        vocab_size=min(max_words, len(tokenizer.word_index) + 1),
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
        num_classes=num_classes
    )
    
    # Print model summary
    model.summary()
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    print("\nTraining BiLSTM model...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    accuracy, y_pred = evaluate_model(model, X_val, y_val, category_mapping)
    
    # Save model artifacts
    save_model_artifacts(model, tokenizer, max_sequence_length, category_mapping)
    
    # Example prediction code
    print("\nExample usage for prediction:")
    print("from tensorflow.keras.models import load_model")
    print("from tensorflow.keras.preprocessing.sequence import pad_sequences")
    print("import pickle")
    print("import numpy as np")
    print("\n# Load model artifacts")
    print("model = load_model('bilstm_news_classifier.h5')")
    print("with open('tokenizer.pickle', 'rb') as handle:")
    print("    tokenizer = pickle.load(handle)")
    print("with open('config.pkl', 'rb') as f:")
    print("    config = pickle.load(f)")
    print("\n# Prepare text for prediction")
    print("text = 'Your news article text here'")
    print("text = text.lower()")  # Basic preprocessing
    print("sequence = tokenizer.texts_to_sequences([text])")
    print("padded_sequence = pad_sequences(sequence, maxlen=config['max_sequence_length'])")
    print("\n# Make prediction")
    print("prediction = model.predict(padded_sequence)[0]")
    print("predicted_class = np.argmax(prediction)")
    print("# Map back to category name")
    print("category_mapping = config['category_mapping']")
    print("index_to_category = {idx: category for category, idx in category_mapping.items()}")
    print("predicted_category = index_to_category[predicted_class]")
    print("print(f'Predicted category: {predicted_category}')")
    print("print(f'Confidence: {prediction[predicted_class]:.4f}')")

if __name__ == "__main__":
    csv_path = "/Users/krutin/Desktop/projects/news classification/backednd/BBC News Train.csv"  # Changed to a generic name for BBC news dataset
    sample_size = None  # Set to None to use all data or an integer to sample
    
    main(csv_path, sample_size)
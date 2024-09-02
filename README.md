
# Recasim-Xonina Detector 🛡️

Welcome to the **Recasim-Xonina Detector** repository! This project is designed to identify and detect comments related to racism and xenophobia. The goal of this project is to leverage machine learning techniques, specifically deep learning, to train a model that can accurately classify comments as either containing or not containing racist or xenophobic content.

## 📁 Project Structure

Here’s a breakdown of the main files included in this repository:

- **Main_Code.ipynb**: This is the core Jupyter notebook where the model development takes place. The notebook covers:
  - **Data Loading**: The notebook begins by loading the labeled dataset into a pandas DataFrame.
  - **Data Preprocessing**: Steps include text cleaning, tokenization, and transforming the text data into a format that can be fed into a machine learning model.
  - **Model Architecture**: The notebook defines a deep learning model using Keras, a high-level neural networks API.
  - **Model Training**: The model is trained on the dataset, with the performance metrics evaluated to fine-tune the model.
  - **Model Evaluation**: Post-training, the model’s performance is analyzed using metrics like accuracy, precision, recall, and confusion matrix.
  - **Saving the Model**: The trained model is saved to a file (`model.keras`) for later use.

- **racism_xenophobia_data_comments_labeled.csv**: This file contains the labeled dataset used for training and evaluating the model. Each row in this CSV file represents a comment along with a label indicating whether the comment contains racist or xenophobic content.

- **Reddit_API.ipynb**: This Jupyter notebook is used for data collection. It interacts with the Reddit API to fetch comments or posts based on certain keywords. The collected data can then be labeled and added to the existing dataset for further training of the model.

- **requirements.txt**: This file lists all the Python packages that are necessary to run the notebooks and train the model. The file includes libraries like TensorFlow, Keras, Pandas, and others required for data processing and model development.

## 🚀 Getting Started

To get started with this project, follow these steps:

### Clone the Repository

First, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/yourusername/Recasim-Xonina-Detector.git
cd Recasim-Xonina-Detector
```

### Set Up the Environment

Ensure you have Python installed on your machine. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

This command will install all the necessary libraries to run the project, including TensorFlow, Keras, Pandas, and others.

## 📝 Usage

### Data Collection

To gather new data, use the `Reddit_API.ipynb` notebook. This notebook allows you to:
- **Authenticate with Reddit's API**: The notebook handles authentication to ensure you can access Reddit's data.
- **Search and Fetch Comments**: Specify keywords related to racism or xenophobia to fetch relevant comments.
- **Store the Data**: The collected data is saved in a CSV format, ready for labeling and further analysis.

### Model Training

The `Main_Code.ipynb` notebook is your go-to for model training. Here's a brief overview of the process:
1. **Load the Dataset**: Load the `racism_xenophobia_data_comments_labeled.csv` file into the notebook.
2. **Preprocess the Data**: Perform necessary preprocessing steps to clean and prepare the data for the model.
3. **Define the Model**: Create a neural network model using Keras.
4. **Train the Model**: Use the prepared dataset to train the model, adjusting parameters as needed.
5. **Evaluate the Model**: Analyze the model's performance to ensure it meets the desired accuracy.

### Model Evaluation and Prediction

After training, the model can be saved and used to predict whether new comments contain racist or xenophobic content. The model can be saved and loaded as follows:

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.keras')

# Predict on new data
predictions = model.predict(new_data)
```

## 📧 Contact

For any questions or assistance, feel free to reach out via email at [shayanebrahimi555@yahoo.com](mailto:your-email@example.com).

---

**Disclaimer**: This project is intended for educational and research purposes, aiming to understand and detect harmful online content. Please use the tool responsibly and adhere to ethical guidelines when handling sensitive data.

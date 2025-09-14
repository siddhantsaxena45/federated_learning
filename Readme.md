
-----

# Federated Learning for Alzheimer's Disease Classification

This project demonstrates a federated learning approach using the Flower framework to train a Convolutional Neural Network (CNN) for classifying stages of Alzheimer's disease. The model is trained on a distributed dataset across two clients without centralizing the data, and the final aggregated model is evaluated for its performance.

-----

## 📈 Project Results

[cite\_start]The final federated model, trained over 3 rounds, was evaluated on the complete, unpartitioned test dataset containing 1283 images[cite: 79, 110].

  * [cite\_start]**Overall Test Accuracy:** **`85.58%`** [cite: 83]

### **Classification Report**

The model shows strong performance, especially for the 'Moderate' and 'Mild' classes. It struggles most with the recall for the 'Very Mild' class, often confusing it with the 'No' dementia class.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Mild | [cite\_start]0.91 [cite: 89] | [cite\_start]0.91 [cite: 90] | [cite\_start]0.91 [cite: 91] | [cite\_start]180 [cite: 92] |
| Moderate | [cite\_start]1.00 [cite: 94] | [cite\_start]1.00 [cite: 95] | [cite\_start]1.00 [cite: 96] | [cite\_start]13 [cite: 97] |
| No | [cite\_start]0.82 [cite: 99] | [cite\_start]0.95 [cite: 100] | [cite\_start]0.88 [cite: 101] | [cite\_start]642 [cite: 102] |
| Very Mild | [cite\_start]0.90 [cite: 104] | [cite\_start]0.69 [cite: 105] | [cite\_start]0.78 [cite: 106] | [cite\_start]448 [cite: 107] |
| **Weighted Avg** | [cite\_start]**0.86** [cite: 116] | [cite\_start]**0.86** [cite: 117] | [cite\_start]**0.85** [cite: 118] | [cite\_start]**1283** [cite: 119] |

### **Confusion Matrix**

The confusion matrix confirms the findings from the classification report. [cite\_start]For instance, it shows that **124** images belonging to the 'Very Mild' class were incorrectly predicted as 'No' dementia[cite: 151].

### **Federated Training Performance**

The global model's accuracy showed consistent improvement with each round of federated training, while the loss decreased.

  * [cite\_start]**Round 1 Accuracy:** \~54.01% [cite: 586]
  * [cite\_start]**Round 2 Accuracy:** \~70.54% [cite: 682]
  * [cite\_start]**Round 3 Accuracy:** \~85.58% [cite: 794]

-----

## 🤖 Model Architecture

The project uses the same simple Convolutional Neural Network (CNN) on both the clients and the server.

  * [cite\_start]**Input Layer:** `(128, 128, 3)` [cite: 214, 331]
  * [cite\_start]**Convolutional Layer 1:** 16 filters, (3,3) kernel, ReLU activation [cite: 215, 332]
  * [cite\_start]**Max Pooling Layer 1** [cite: 216, 333]
  * [cite\_start]**Convolutional Layer 2:** 32 filters, (3,3) kernel, ReLU activation [cite: 217, 334]
  * [cite\_start]**Max Pooling Layer 2** [cite: 218, 335]
  * [cite\_start]**Flatten Layer** [cite: 219, 336]
  * [cite\_start]**Dense Layer 1:** 64 units, ReLU activation [cite: 220, 337]
  * [cite\_start]**Output Layer:** 4 units, Softmax activation [cite: 221, 338]

[cite\_start]The model is compiled using the `adam` optimizer and `sparse_categorical_crossentropy` loss function[cite: 223].

-----

## 📂 Project Structure

This project consists of three main Python scripts:

  * [cite\_start]`server.py`: Initializes the Flower server using a `FedAvg` strategy[cite: 376]. [cite\_start]It runs for 3 rounds [cite: 417][cite\_start], aggregates model weights, evaluates metrics, and saves the final global model as `federated_model.h5`[cite: 429].
  * `client.py`: Connects to the Flower server. [cite\_start]It loads a unique partition of the "AlzheimerDataset"[cite: 194], trains the CNN on its local data, and sends the updated weights back to the server.
  * [cite\_start]`test_model.py`: A standalone script to load the final `federated_model.h5` [cite: 11, 20] [cite\_start]and evaluate its performance on the entire, unpartitioned test dataset[cite: 22].

-----

## 🚀 Setup and Execution

Follow these steps to replicate the experiment.

### **1. Prerequisites**

  * Python 3.8+
  * [cite\_start]The "AlzheimerDataset" directory available with `train` and `test` subdirectories[cite: 178, 183].

### **2. Installation**

Create a virtual environment and install the required packages.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn "flwr[simulation]"
```

### **3. Running the Experiment**

You will need to open **three separate terminal windows**. Make sure you activate the virtual environment in each one.

**Terminal 1: Start the Server**

```bash
python server.py
```

[cite\_start]The server will start and wait for clients to connect[cite: 412].

**Terminal 2: Start Client 0**

```bash
python client.py
```

When prompted, enter the client ID:

```
Enter client ID (0 or 1): 0
```

**Terminal 3: Start Client 1**

```bash
python client.py
```

When prompted, enter the client ID:

```
Enter client ID (0 or 1): 1
```

The federated training process will now begin. After 3 rounds, the server will automatically save the final model.

### **4. Evaluate the Final Model**

Once the server has finished and saved `federated_model.h5`, run the evaluation script in a new terminal.

```bash
python test_model.py
```

[cite\_start]This will print the final accuracy, classification report, and display the confusion matrix for the globally aggregated model[cite: 43, 60].


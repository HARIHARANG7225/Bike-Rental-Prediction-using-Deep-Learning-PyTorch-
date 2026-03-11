# 🚴 Bike Rental Prediction using Deep Learning (PyTorch)

## 📌 Project Overview
This project applies **Deep Learning with PyTorch** to predict hourly bike rental demand using the **Bike Sharing Dataset**.  
The goal is to model demand patterns based on **seasonality, holidays, weather conditions, and user registrations**.

---

## ⚙️ Tech Stack
- **Python**: pandas, numpy, scikit-learn, seaborn, matplotlib
- **Deep Learning**: PyTorch
- **Data Handling**: train-test split, DataLoader batching
- **Evaluation**: RMSE (Root Mean Squared Error)

---

## 📊 Workflow
1. **Data Exploration**
   - Visualized seasonal and holiday trends
   - Histogram and bar plots for demand distribution

2. **Feature Engineering**
   - One-hot encoding for categorical variables (season, weather)
   - Normalized numerical features

3. **Model Architecture**
   - Input Layer: 11 features
   - Hidden Layer: 10 neurons + Dropout (p=0.2)
   - Output Layer: 1 (predicted rental count)

4. **Training**
   - Loss Function: MSELoss
   - Optimizer: (to be added, e.g., Adam/SGD)
   - Epochs: 350
   - Batch Size: 100

5. **Evaluation**
   - RMSE calculated on test set
   - Demonstrated predictive accuracy on unseen data

---

## 📈 Results
- The model successfully captured demand patterns.
- RMSE score validated the model’s effectiveness.

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/bike-rental-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook bike_rental_prediction.ipynb

🔮 Future Improvements
- Hyperparameter tuning (learning rate, optimizer choice)
- Adding more hidden layers for deeper representation
- Experimenting with ReLU activation for non-linear learning

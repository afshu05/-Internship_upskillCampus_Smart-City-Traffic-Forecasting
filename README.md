🚦 Smart City Traffic Pattern Forecasting using Machine Learning

📌 Project Overview
This project focuses on analyzing and forecasting traffic patterns in a smart city environment. The goal is to help government authorities understand traffic flow across different junctions and prepare for peak traffic conditions.

Using machine learning techniques, the system predicts future traffic based on historical data, enabling better traffic management and infrastructure planning.

## 🎯 Objectives
- Analyze traffic data from multiple junctions
- Identify traffic patterns and peak hours
- Build a machine learning model for prediction
- Support smart city traffic management

📊 Dataset
The dataset is taken from Kaggle and contains:
- DateTime (date and time)
- Junction (traffic junction ID)
- Vehicles (traffic count)

Files used:
- `train_aWnotuB.csv`
- `test_BdBKkAj.csv`

 ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

🧠 Machine Learning Model
- Algorithm: Random Forest Regressor
- Features Used:
  - Hour
  - Day
  - Month
  - Year
  - DayOfWeek
  - Weekend
  - Junction

🔄 Project Workflow
1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Prediction

📈 Output
- Model accuracy and RMSE
- Graph showing Actual vs Predicted traffic
- Predicted traffic values
- `submission.csv` file

▶️ How to Run

1. Install dependencies:

pip install pandas numpy matplotlib scikit-learn


2. Place files in same folder:
- train_aWnotuB.csv
- test_BdBKkAj.csv
- main.py

3. Run the project:
python main.py

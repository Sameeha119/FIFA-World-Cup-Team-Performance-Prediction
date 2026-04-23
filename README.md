# FIFA-World-Cup-Team-Performance-Prediction
This project uses machine learning to predict the **goals scored per match (goals_per90)** by football teams based on performance statistics such as possession, passing accuracy, and defensive actions.

---

## 📌 Project Overview

The goal of this project is to:

* Analyze team performance data
* Build predictive models
* Compare different machine learning algorithms
* Identify which teams overperform or underperform expectations

---

## 📊 Dataset

The dataset contains **team-level football statistics**, including:

* Possession (%)
* Expected goals (xG)
* Shots on target
* Passing accuracy
* Defensive metrics (tackles, interceptions)
* Goalkeeping stats

Total features: **189 columns**

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas & NumPy (Data processing)
* Scikit-learn (Machine Learning)
* Matplotlib (Visualization)

---

## 🧠 Models Used

Three regression models were trained and evaluated:

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

---

## 🏗️ Workflow

1. Load dataset
2. Clean and inspect data
3. Select important features:

   * possession
   * assists_per90
   * xg_per90
   * shots_on_target_per90
   * passes_pct
   * tackles
   * interceptions
   * gk_save_pct
4. Split data into training and testing sets
5. Train models
6. Evaluate performance using:

   * Mean Absolute Error (MAE)
   * R² Score
7. Generate predictions
8. Visualize results

---

## 📈 Model Performance

| Model             | MAE   | R² Score |
| ----------------- | ----- | -------- |
| Linear Regression | 0.213 | 0.51 ✅   |
| Decision Tree     | 0.346 | -0.72 ❌  |
| Random Forest     | 0.230 | 0.37 ⚠️  |

👉 **Best Model: Linear Regression**

---

## 🔮 Predictions

The model predicts goals per match for each team.

### Top Predicted Teams:

* England
* Portugal
* Germany
* France

---

## 📉 Overperformance Analysis

We calculated:

```
difference = actual goals_per90 - predicted_goals
```

### Teams that overperformed:

* Ghana
* Spain
* Senegal

### Teams that underperformed:

* Croatia
* Canada
* Mexico

---

## 📊 Visualization

A horizontal bar chart was created to show **predicted goals per team**, making it easy to compare team performance visually.


---

## 📌 Key Insights

* Possession and xG are strong predictors of goals
* Simple models like Linear Regression can outperform complex ones
* Some teams significantly outperform statistical expectations

---

## ⚠️ Limitations

* Small dataset (limited number of teams)
* No player-level data
* No match-by-match data (only aggregated stats)
* Model may not generalize well to other competitions

---

## 📂 Project Structure

```
FIFA-World-Cup-Team-Performance-Prediction
│
├── data
│   └── team_data_data.csv
│
├── notebooks
│   └── FIFA World Cup Team Performance Prediction.ipynb
│
└── README.md
```

## 💡 Future Improvements

* Add more data (multiple tournaments/seasons)
* Use advanced models (XGBoost, Neural Networks)
* Include player-level features
* Perform feature engineering
* Hyperparameter tuning

---

## 👩‍💻 Author

[Simiat Ahmed](linkedin.com/in/simiat-ahmed-bbbb58146/)

---

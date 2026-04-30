# Introduction

Football analytics has become an important tool for understanding team performance and predicting outcomes. By using data analysis and machine learning techniques, we can identify patterns in team statistics and estimate future performance.

In this project, a dataset containing various football team statistics was analyzed to predict the number of goals scored per 90 minutes. Several performance metrics such as possession, assists, expected goals, shots on target, and defensive actions were used as predictors.

Different machine learning models were applied to evaluate how well these statistics can predict team scoring performance. The results help identify which teams are likely to score more goals based on their playing style and match statistics.

# Objectives

The main objectives of this project are:

1. To analyze football team performance data using Python.
2. To identify important features that influence goal scoring.
3. To build predictive models that estimate goals scored per 90 minutes.
4. To compare different machine learning algorithms and evaluate their performance.
5. To predict which teams are likely to score more goals based on their statistics.

# Explanation of Each Step 

### Step 1: Importing Libraries


```python
import pandas as pd
import numpy as np
```

At the beginning, Python libraries were imported.

* **Pandas** was used to load and manipulate the dataset.
* **NumPy** was used for numerical operations and handling arrays.

These libraries are commonly used in data analysis and machine learning.

###  Step 2: Loading the Dataset


```python
df = pd.read_csv(r"C:\Users\DELL\Downloads\world cup data 2\team_data.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Teams</th>
      <th>Continent</th>
      <th>players_used</th>
      <th>avg_age</th>
      <th>possession</th>
      <th>games</th>
      <th>games_starts</th>
      <th>minutes</th>
      <th>minutes_90s</th>
      <th>goals</th>
      <th>...</th>
      <th>fouls</th>
      <th>fouled</th>
      <th>offsides</th>
      <th>pens_won</th>
      <th>pens_conceded</th>
      <th>own_goals</th>
      <th>ball_recoveries</th>
      <th>aerials_won</th>
      <th>aerials_lost</th>
      <th>aerials_won_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Argentina</td>
      <td>South America</td>
      <td>24</td>
      <td>28.4</td>
      <td>57.4</td>
      <td>7</td>
      <td>77</td>
      <td>690</td>
      <td>7.7</td>
      <td>15</td>
      <td>...</td>
      <td>100</td>
      <td>115</td>
      <td>23</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>357</td>
      <td>83</td>
      <td>90</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>Asia</td>
      <td>20</td>
      <td>28.7</td>
      <td>37.8</td>
      <td>4</td>
      <td>44</td>
      <td>360</td>
      <td>4.0</td>
      <td>3</td>
      <td>...</td>
      <td>52</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>72</td>
      <td>72</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Belgium</td>
      <td>Europe</td>
      <td>20</td>
      <td>30.6</td>
      <td>57.0</td>
      <td>3</td>
      <td>33</td>
      <td>270</td>
      <td>3.0</td>
      <td>1</td>
      <td>...</td>
      <td>30</td>
      <td>35</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>132</td>
      <td>33</td>
      <td>28</td>
      <td>54.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>South America</td>
      <td>26</td>
      <td>28.5</td>
      <td>56.2</td>
      <td>5</td>
      <td>55</td>
      <td>480</td>
      <td>5.3</td>
      <td>8</td>
      <td>...</td>
      <td>63</td>
      <td>74</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>271</td>
      <td>43</td>
      <td>56</td>
      <td>43.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cameroon</td>
      <td>Africa</td>
      <td>22</td>
      <td>28.0</td>
      <td>41.7</td>
      <td>3</td>
      <td>33</td>
      <td>270</td>
      <td>3.0</td>
      <td>4</td>
      <td>...</td>
      <td>32</td>
      <td>38</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>42</td>
      <td>36</td>
      <td>53.8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 191 columns</p>
</div>



The dataset containing football team statistics was loaded using Pandas.

The **head()** function was used to display the first five rows of the dataset. This step helps to:

* understand the structure of the data
* check the available columns
* confirm that the dataset loaded correctly.

### Step 3: Checking for Missing Values


```python
df.isnull().sum()
```




    Teams              0
    Continent          0
    players_used       0
    avg_age            0
    possession         0
                      ..
    own_goals          0
    ball_recoveries    0
    aerials_won        0
    aerials_lost       0
    aerials_won_pct    0
    Length: 191, dtype: int64



This step checks whether there are missing values in the dataset. Missing values can negatively affect machine learning models.

The results showed that there were **no missing values**, which means the dataset was clean and ready for analysis.

### Step 4: Exploring the Dataset Columns


```python
print(df.columns.tolist())
```

    ['Teams', 'Continent', 'players_used', 'avg_age', 'possession', 'games', 'games_starts', 'minutes', 'minutes_90s', 'goals', 'assists', 'goals_pens', 'pens_made', 'pens_att', 'cards_yellow', 'cards_red', 'Total cards', 'goals_per90', 'assists_per90', 'goals_assists_per90', 'goals_pens_per90', 'goals_assists_pens_per90', 'xg', 'npxg', 'xg_assist', 'npxg_xg_assist', 'xg_per90', 'xg_assist_per90', 'xg_xg_assist_per90', 'npxg_per90', 'npxg_xg_assist_per90', 'gk_games', 'gk_games_starts', 'gk_minutes', 'gk_goals_against', 'gk_goals_against_per90', 'gk_shots_on_target_against', 'gk_saves', 'gk_save_pct', 'gk_wins', 'gk_ties', 'gk_losses', 'gk_clean_sheets', 'gk_clean_sheets_pct', 'gk_pens_att', 'gk_pens_allowed', 'gk_pens_saved', 'gk_pens_missed', 'gk_pens_save_pct', 'gk_free_kick_goals_against', 'gk_corner_kick_goals_against', 'gk_own_goals_against', 'gk_psxg', 'gk_psnpxg_per_shot_on_target_against', 'gk_psxg_net', 'gk_psxg_net_per90', 'gk_passes_completed_launched', 'gk_passes_launched', 'gk_passes_pct_launched', 'gk_passes', 'gk_passes_throws', 'gk_pct_passes_launched', 'gk_passes_length_avg', 'gk_goal_kicks', 'gk_pct_goal_kicks_launched', 'gk_goal_kick_length_avg', 'gk_crosses', 'gk_crosses_stopped', 'gk_crosses_stopped_pct', 'gk_def_actions_outside_pen_area', 'gk_def_actions_outside_pen_area_per90', 'gk_avg_distance_def_actions', 'shots', 'shots_on_target', 'shots_on_target_pct', 'shots_per90', 'shots_on_target_per90', 'goals_per_shot', 'goals_per_shot_on_target', 'average_shot_distance', 'shots_free_kicks', 'npxg_per_shot', 'xg_net', 'npxg_net', 'passes_completed', 'passes', 'passes_pct', 'passes_total_distance', 'passes_progressive_distance', 'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium', 'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'pass_xa', 'xg_assist_net', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes', 'passes_live', 'passes_dead', 'passes_free_kicks', 'through_balls', 'passes_switches', 'crosses', 'throw_ins', 'corner_kicks', 'corner_kicks_in', 'corner_kicks_out', 'corner_kicks_straight', 'passes_offsides', 'passes_blocked', 'sca', 'sca_per90', 'sca_passes_live', 'sca_passes_dead', 'sca_dribbles', 'sca_shots', 'sca_fouled', 'sca_defense', 'gca', 'gca_per90', 'gca_passes_live', 'gca_passes_dead', 'gca_dribbles', 'gca_shots', 'gca_fouled', 'gca_defense', 'tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'dribble_tackles', 'dribbles_vs', 'dribble_tackles_pct', 'dribbled_past', 'blocks', 'blocked_shots', 'blocked_passes', 'interceptions', 'tackles_interceptions', 'clearances', 'errors', 'touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'touches_live_ball', 'dribbles_completed', 'dribbles', 'dribbles_completed_pct', 'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received', 'minutes_per_game', 'minutes_pct', 'minutes_per_start', 'games_complete', 'games_subs', 'minutes_per_sub', 'unused_subs', 'points_per_game', 'on_goals_for', 'on_goals_against', 'plus_minus', 'plus_minus_per90', 'on_xg_for', 'on_xg_against', 'xg_plus_minus', 'xg_plus_minus_per90', 'cards_yellow_red', 'fouls', 'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']
    

This step lists all the column names in the dataset.

The dataset contains many football statistics such as:

* possession
* goals
* assists
* shots
* passes
* tackles
* interceptions
* goalkeeper statistics

These variables represent different aspects of team performance.

### Step 5: Importing Machine Learning Models


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
```

##### Several machine learning tools from Scikit-Learn were imported:

* **train_test_split** – to divide the dataset into training and testing sets
* **mean_absolute_error (MAE)** – to measure prediction error
* **R² score** – to measure how well the model explains the data

Three models were used:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Step 6: Selecting Important Features


```python
features = [
    'possession',
    'assists_per90',
    'xg_per90',
    'shots_on_target_per90',
    'passes_pct',
    'tackles',
    'interceptions',
    'gk_save_pct'
]

X = df[features]
y = df['goals_per90']
```

**Important features that influence goal scoring were selected.** 
These include:

* possession percentage
* assists per 90 minutes
* expected goals per 90 minutes
* shots on target per 90 minutes
* passing accuracy
* tackles
* interceptions
* goalkeeper save percentage

These variables were used as **input features** for the prediction model.

### Step 7: Defining Input and Target Variables

* **X (features)** represents the input variables used to make predictions.
* **y (target)** represents the output variable, which is **goals scored per 90 minutes**.


```python
X = df[features]
y = df['goals_per90']
```

### Step 8: Splitting the Data


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**The dataset was divided into two parts:**

* **Training data (80%)** – used to train the model
* **Testing data (20%)** – used to evaluate the model performance

This helps ensure that the model works well on unseen data.

### Step 9: Training the Linear Regression Model, Making Predictions and Evaluating the Model


```python
lr = LinearRegression()
lr.fit(X_train, y_train)

preds_lr = lr.predict(X_test)

print("Linear Regression MAE:", mean_absolute_error(y_test, preds_lr))
print("Linear Regression R2:", r2_score(y_test, preds_lr))
```

    Linear Regression MAE: 0.21259612466129255
    Linear Regression R2: 0.5117056150629108
    

The Linear Regression model was trained using the training dataset.

This model learns the relationship between the selected features and the number of goals scored.
The trained model was used to predict goals on the testing dataset.

Two evaluation metrics were used:

* **MAE (Mean Absolute Error)** measures the average prediction error.
* **R² Score** shows how well the model explains the variation in the data.

### Step 10: Training Additional Models

Two additional models were trained:

**Decision Tree**


```python
dt = DecisionTreeRegressor(max_depth=4)
dt.fit(X_train, y_train)

preds_dt = dt.predict(X_test)

print("Decision Tree MAE:", mean_absolute_error(y_test, preds_dt))
print("Decision Tree R2:", r2_score(y_test, preds_dt))
```

    Decision Tree MAE: 0.29333333333333333
    Decision Tree R2: -0.421471827277514
    

This model creates a tree-like structure of decisions to predict outcomes.

**Random Forest**


```python
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

preds_rf = rf.predict(X_test)

print("Random Forest MAE:", mean_absolute_error(y_test, preds_rf))
print("Random Forest R2:", r2_score(y_test, preds_rf))
```

    Random Forest MAE: 0.22974285714285703
    Random Forest R2: 0.3670203896013384
    

This model combines multiple decision trees to improve prediction accuracy.

Their performance was compared with the Linear Regression model.


```python
print("LR MAE:", mean_absolute_error(y_test, preds_lr))
print("DT MAE:", mean_absolute_error(y_test, preds_dt))
print("RF MAE:", mean_absolute_error(y_test, preds_rf))
```

    LR MAE: 0.21259612466129255
    DT MAE: 0.29333333333333333
    RF MAE: 0.22974285714285703
    

The Linear Regression model achieved the best performance compared to the other models.

### Step 11: Predicting Goals for Each Team


```python
df['predicted_goals'] = lr.predict(X)
```

After selecting the best model (Linear Regression), predictions were generated for all teams in the dataset.

### Step 12: Ranking Teams by Predicted Goals


```python
df[['Teams', 'predicted_goals']].sort_values(
    by='predicted_goals', ascending=False
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Teams</th>
      <th>predicted_goals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>England</td>
      <td>2.544829</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Portugal</td>
      <td>2.513726</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Germany</td>
      <td>2.262176</td>
    </tr>
    <tr>
      <th>11</th>
      <td>France</td>
      <td>2.061973</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Serbia</td>
      <td>1.867321</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Argentina</td>
      <td>1.789339</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Netherlands</td>
      <td>1.645017</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Spain</td>
      <td>1.573907</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croatia</td>
      <td>1.516978</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>1.497313</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cameroon</td>
      <td>1.401749</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Switzerland</td>
      <td>1.390267</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecuador</td>
      <td>1.184802</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Iran</td>
      <td>1.166530</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Japan</td>
      <td>1.124216</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ghana</td>
      <td>1.122526</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Korea Republic</td>
      <td>1.068097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>1.002378</td>
    </tr>
    <tr>
      <th>29</th>
      <td>United States</td>
      <td>0.964611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Saudi Arabia</td>
      <td>0.957215</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Mexico</td>
      <td>0.945235</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Morocco</td>
      <td>0.938989</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Senegal</td>
      <td>0.851967</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Costa Rica</td>
      <td>0.728175</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Canada</td>
      <td>0.710335</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Tunisia</td>
      <td>0.592141</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Qatar</td>
      <td>0.549780</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Denmark</td>
      <td>0.488972</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Belgium</td>
      <td>0.439627</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Uruguay</td>
      <td>0.399661</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Poland</td>
      <td>0.394605</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Wales</td>
      <td>0.365007</td>
    </tr>
  </tbody>
</table>
</div>


Teams were ranked based on their predicted goals per 90 minutes.

This allows us to identify which teams are expected to score the most goals according to their statistics.

```python
df['difference'] = df['goals_per90'] - df['predicted_goals']

df[['Teams', 'difference']].sort_values(
    by='difference', ascending=False
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Teams</th>
      <th>difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13</th>
      <td>Ghana</td>
      <td>0.547474</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Spain</td>
      <td>0.506093</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Senegal</td>
      <td>0.398033</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Poland</td>
      <td>0.355395</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Costa Rica</td>
      <td>0.271825</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Uruguay</td>
      <td>0.270339</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Netherlands</td>
      <td>0.224983</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Korea Republic</td>
      <td>0.181903</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Argentina</td>
      <td>0.170661</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Iran</td>
      <td>0.163470</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ecuador</td>
      <td>0.145198</td>
    </tr>
    <tr>
      <th>11</th>
      <td>France</td>
      <td>0.118027</td>
    </tr>
    <tr>
      <th>10</th>
      <td>England</td>
      <td>0.055171</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Saudi Arabia</td>
      <td>0.042785</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Japan</td>
      <td>0.025784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>0.002687</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Wales</td>
      <td>-0.035007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cameroon</td>
      <td>-0.071749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Belgium</td>
      <td>-0.109627</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Portugal</td>
      <td>-0.113726</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Morocco</td>
      <td>-0.118989</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Switzerland</td>
      <td>-0.140267</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Denmark</td>
      <td>-0.158972</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Serbia</td>
      <td>-0.197321</td>
    </tr>
    <tr>
      <th>29</th>
      <td>United States</td>
      <td>-0.214611</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Qatar</td>
      <td>-0.219780</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>-0.252378</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Tunisia</td>
      <td>-0.262141</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Germany</td>
      <td>-0.262176</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Mexico</td>
      <td>-0.275235</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Canada</td>
      <td>-0.380335</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croatia</td>
      <td>-0.476978</td>
    </tr>
  </tbody>
</table>
</div>



# Results and Findings

After training and evaluating the models, **the Linear Regression model produced the best results**.

**Linear Regression** had the **lowest prediction error (MAE ≈ 0.21)**.
It also had the **highest R² score (≈ 0.51)** compared to the other models.

This indicates that Linear Regression was better at predicting goals per 90 minutes using the selected features.

Using this model, predicted goals were calculated for each team and ranked from highest to lowest.

The results showed that teams such as:

* England
* Portugal
* Germany
* France

had the highest predicted goal-scoring performance based on their statistics.

### Step 13: Visualizing Predicted Goals per Team 
The Matplotlib library was imported to create charts and graphs.
It is commonly used in Python to visualize data and make results easier to understand.

The dataset was sorted based on the predicted goals column in descending order.
This means:

* Teams with higher predicted goals appear first
* Teams with lower predicted goals appear last

Sorting makes the visualization clearer and easier to interpret.

A **horizontal bar chart** was created.

* The **y-axis** shows the team names.
* The **x-axis** shows the predicted number of goals per match.

Each bar represents a team, and the length of the bar shows how many goals the model predicts that team will score.



```python
import matplotlib.pyplot as plt

df_sorted = df.sort_values(by='predicted_goals', ascending=False)

plt.figure()
plt.barh(df_sorted['Teams'], df_sorted['predicted_goals'])
plt.title("Predicted Goals per Team")
plt.xlabel("Goals per Match")
plt.ylabel("Team")
plt.gca().invert_yaxis()
plt.show()
```


    
![png](output_46_0.png)
    


The chart shows the **predicted goals per match for each team** based on the machine learning model. The predicted values range from **about 2.54 goals per match to about 0.36 goals per match.**

Teams such as **England (2.54), Portugal (2.51), Germany (2.26), and France (2.06)** have the highest predicted goals, which suggests strong attacking performance.

Several teams, including **Argentina, Netherlands, Spain, Croatia, and Brazil**, fall in the middle range with **about 1 to 1.8 goals per match**. 

On the other hand, teams such as **Qatar, Denmark, Belgium, Uruguay, Poland, and Wales** have predicted values **below 1 goal per match**, indicating weaker scoring potential based on the model.

# Limitations

Although the model produced useful predictions, several limitations should be considered.

* #### Small Dataset

The dataset only contains statistics for a limited number of teams, which may reduce the accuracy of the machine learning models.

* #### Match Differences

Some teams played more matches than others. This can affect cumulative statistics such as tackles, passes, or fouls.

* #### Limited Features

The model only used a small number of selected features. Other important factors such as player injuries, tactics, team chemistry, and match conditions were not included.

*  #### Model Simplicity

Linear Regression assumes a simple relationship between variables, but football performance can be influenced by more complex interactions.

# Conclusion

This project demonstrated how football performance data can be analyzed using machine learning techniques to predict goal scoring. By examining various team statistics such as possession, assists, expected goals, and defensive actions, predictive models were developed to estimate goals per 90 minutes.

Among the tested models, Linear Regression performed the best based on evaluation metrics. The model was able to identify teams that are more likely to score goals based on their playing statistics.

Overall, the project shows how data analytics can help understand team performance and support decision-making in football analysis.

<div style="font-family: sans-serif; padding: 10px 0;">
  <a href="https://www.linkedin.com/in/simiat-ahmed-bbbb58146/" 
     style="color: #0077b5; 
            text-decoration: none; 
            font-weight: 600; 
            border-bottom: 2px solid #0077b5; 
            padding-bottom: 2px;">
    Simiat Ahmed
  </a>
</div>

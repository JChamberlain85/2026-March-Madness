# 2026 March Madness Bracket Predictor

This repository contains a machine learning pipeline designed to predict the outcomes of the 2026 Men's NCAA Basketball Tournament. Using historical Kaggle datasets and advanced basketball analytics, the program trains an Extreme Gradient Boosting (XGBoost) model to forecast the point differential between any two matched teams, automatically advancing the winners through a simulated 64-team bracket.
## Program Scope and Execution

Extreme Gradient Boosting (XGBoost) is an ensemble machine learning algorithm that builds a series of decision trees sequentially, where each new tree is specifically designed to predict and correct the residual errors made by the previous ones. By continuously learning from past mistakes and using gradient descent to minimize its loss function, it gradually converges on highly accurate forecasts. 

For predicting March Madness point spreads, XGBoost was the optimal choice because it excels at capturing complex, non-linear interactions within tabular data—such as how a team's pace might uniquely interact with an opponent's rebounding or turnover metrics. College basketball data is notoriously volatile, filled with garbage-time scoring, intentional fouling, and unpredictable upsets. XGBoost addresses this through its robust, built-in mathematical regularization. By combining these regularization techniques with conservative, shallow decision trees, the model is prevented from overfitting to the "noise" of the tournament. Instead of memorizing historical anomalies, it isolates the true underlying statistical advantages between two matchups, making it exceptionally well-suited for the high-variance environment of NCAA basketball.


## 1. Data Aggregation and Feature Engineering

Season and Tournament data ranging 2003-present from Kaggle's Men's March Madness Database was utilized to calculate advanced metrics listed below. 

    Pace & Efficiency: Estimated Possessions, Offensive/Defensive Efficiency (Points per Possession), and Net Rating.

    Shooting Metrics: Effective Field Goal Percentage (eFG%), True Shooting Percentage (TS%), Free Throw Rate (FT%), and 3-Point Attempt Rate (3PA%).

    Ball Control: Turnover Percentage (TO%), Assist Percentage (Ast%), and Assist-to-Turnover Ratio.

    Defense & Hustle: Offensive/Defensive Rebound Percentages (ORb%, DRb%), Steal Percentage (Stl%), and Block Percentage (Blk%).

## 2. Team Averages & Matchup Differentials

Once game-level stats are calculated, the code groups the data by Season and TeamID to calculate season-long averages.

To prep the data for the model, the script pairs teams up and calculates the statistical differences between Team 1 and Team 2 (e.g., Team 1 Net Rating - Team 2 Net Rating, Team 1 Seed - Team 2 Seed). To prevent the model from learning an arbitrary ordering bias (e.g., assuming "Team 1" is always the favorite), the team assignments are randomized before computing the differentials.
## How the Model Works

Rather than a simple binary win/loss classification, this program uses an XGBoost Regressor to predict the exact point spread of the game. This allows us to grasp the confidence the model has in its prediction.

## Model Architecture

    Algorithm: XGBRegressor utilizing gradient-boosted decision trees.

    Validation: 10-Fold Cross-Validation ensures the model is evaluated on multiple distinct subsets of the data to prevent overfitting.

    Hyperparameters: Tuned with a shallow depth (max_depth=3), 
                     a conservative learning rate (eta=0.015), 
                     and heavy regularization (reg_lambda=5, min_child_weight=3) 
                     to prevent the model from overfitting.

## Model Prediction Accuracy

March Madness is notoriously hard to predict, with 63 games over the course of the tournament and a probability of 100% accuracy at 1 in 9.22e^18 (2^63). The model performed admirably well with 49/63 correct predictions or 77.78% accuracy. 

Just by dumb luck a bracket prediction can be extremely accurate, however I had already determined the model would be great at its job just based on the initial testing data. 

Over the 10-fold cross-validation process, the model yielded the following error metrics:

    Average Mean Absolute Error (MAE): 8.67 ± 0.50 points
    Average Root Mean Squared Error (RMSE): 10.78 ± 0.60 points
    
Meaning that on average, the model's predicted point differential is within about 8.7 points of the actual final score. Given the high variance of single-elimination college basketball (where late-game intentional fouling and blowout garbage time heavily impact final scores), an MAE under 9 points represents a highly competitive baseline for spread betting and bracketology.
## 2026 Tournament Prediction Results

The script simulates the entire 2026 tournament round-by-round, feeding the winners of each round into the next. According to the XGBoost model's season-average differential calculations, here is how the model performed each round:

| Round                  | Correct Pred | Matches  | % Accuracy |
|------------------------|--------------|----------|------------|
| Round of 64            | 28           | 32       | 87.5%      |
| Round of 32            | 12           | 16       | 75%        |
| Sweet 16               | 5            | 8        | 62.5%      |
| Elite 8                | 2            | 4        | 50%        |
| Final Four             | 1            | 2        | 50%        |
| National Championship  | 1            | 1        | 100%       |
| **Total**              | **49**           | **63**       | **77.78%**     |

## Comparison to Other Brackets

While this wasn't the #1 bracket or even the best designed model it did destroy the competition I had initially set out for it:
(Points are based on the round with points per correct pick increasing each round)

| Bracket                  | Points | % Accuracy | Type                 |
|--------------------------|--------|------------|----------------------|
|  **XGBoost Model**       | **1360** | **77.8%** | **XGBoost**         |
| ESPN Auto Picks          | 870    | 71.4%      | Algorithmic          |
| My Picks                 |  760   | 66.7%    |   Hand Picked         | 
| ChatGPT Picks            | 740    | 63.5%      | LLM-based logic      |
| **My 2025 ML Model Picks**           | **660**    | **22.9%**      | **Neural Network**       |
| Random Picks             | 310    | 7.2%       | Pure RNG             |


## Conclusion

My 2026 March Madness Bracket Predictor successfully demonstrates the power of machine learning in navigating one of the most volatile events in sports. By leveraging advanced basketball analytics and the robust, noise-filtering capabilities of an XGBoost Regressor, the model achieved a highly impressive 77.78% overall accuracy. Correctly predicting 49 out of 63 games—including the National Champion—proves that focusing on underlying statistical advantages and exact point differentials is a highly effective strategy.

The model's performance metrics further validate its structural integrity. An Average Mean Absolute Error (MAE) of 8.67 points is a strong baseline for college basketball, indicating that the algorithm's conservative learning rate and heavy regularization successfully prevented it from overfitting to unpredictable anomalies or garbage-time scoring.

Most importantly, the model absolutely dismantled the competition, I've learned a lot about machine learning and statistical methods since March of 2025 and hopefully next year I can show the same year over year performance increase.

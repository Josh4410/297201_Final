import streamlit as st

# Scrapping Libraries
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup # Creates structured, searchable object
# Files and basic libraries
import time
from io import StringIO
import os
import json
#Dataframe Libraries
import numpy as np
import pandas as pd
# Visualisation Librarries
import matplotlib.pyplot as plt
import seaborn as sns
#Stats Modelling
from pylab import rcParams
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# tit;es
st.set_page_config(page_title="Keepers Table!")
st.title("Is La Liga a more intense league to watch?")
st.write("Analysis of goal keepers data from Premier League and La Liga across the 3 latest seasons (2021-2024)")

st.write("More red cards would suggest a intense game?")
st.link_button("Red Cards per League 2019-2022", "https://static01.nyt.com/athletic/uploads/wp/2024/10/16144823/league_fouls.png")

### --------------------------------------------- READ IN DATA ---------------------------------------------
keeper_combined = pd.read_csv('keeper_combined.csv')
st.write("Head of the data we scrapped from: ")
st.link_button("fbref.com","https://fbref.com/en/")
st.dataframe(keeper_combined.head())
st.write(f"Shape: {keeper_combined.shape[0]} rows, {keeper_combined.shape[1]} columns")

### --------------------------------------------- QUICK CLEAN ---------------------------------------------
#keeper_combined['90s'] = keeper_combined['90s'].str.strip() can't use str.strip because now this column is being read as a float! has been a nightmare :/
keeper_combined['90s'] = pd.to_numeric(keeper_combined['90s'], errors='coerce')
#Since the column doesn't contin any valuable data, I'm just dropping it and moving on
keeper_combined = keeper_combined.drop('90s', axis=1)

new_column_names = ['Squad','Num_Goalkeepers','Goals_Against','Penalty_Kicks_Allowed','Free_Kick_Goal_Against','Corner_Kick_Goals_Against','Own_Goals',                         
    #Expected
    'EXP_Post_Shot_Goals','EXP_Post_Shot_per_Shot_on_Target','EXP_Post_Shot_Minus_Goals_Allowed','EXP_Post_Shot_Minus_Goals_Allowed_Per_90_Minutes',
    #Launched
    'Passes_Completed_Over_40yrds','Passes_Attempted_Over_40yrds','Pass_Completion_percent_Over_40yrds',             
    #Passes
    'Passes_Attempted','Throws_Attempted','Launched_Passes_percent','Average_Length_Pass',                     
    #Goal Kicks
    'Goal_kicks_Attempted','Goal_kicks_Launched_percent','Goal_Kicks_Average_Length',                  
    #Crosses
    'Crosses_Faced','Crosses_Stopped','Crosses_Stopped_percent',                        
    #Sweeper
    'Num_Defensive_Actions_Outside_PA','Num_Defensive_Actions_Outside_PA_Per_90','Avg_Def_Dist_From_Goal','League','Season']
keeper_combined.columns = new_column_names
keeper_combined['Premier League'] = keeper_combined['League'].map({'La Liga': 0, 'Premier League': 1})
keeper_combined['La Liga'] = keeper_combined['League'].map({'La Liga': 1, 'Premier League': 0})
keeper_combined.rename(columns={'Premier League': 'Premier_League', 'La Liga': 'La_Liga'}, inplace=True)

numeric_df = keeper_combined.apply(pd.to_numeric, errors='coerce').convert_dtypes() #convert keeper_combined to numeric
numeric_df = numeric_df.drop(['Season','League'], axis=1) #drop non numeric columns

### --------------------------------------------- CORRELATION MATRIX ---------------------------------------------
st.markdown("## What features can we notice just from visualising the data?")
#Create a correlation matrix of only the numeric data
numeric_df = numeric_df.select_dtypes(include='number') # select only numeric data
corr_matrix = numeric_df.corr() #Correlation Matrix
fig, ax = plt.subplots(figsize=(13,13)) #has to be big so you can see the numbers   
sns.heatmap(corr_matrix, annot=True, cmap='YlOrBr', fmt=".1f")
ax.set_title('Keepers Correlation Heatmap')
st.pyplot(fig)

# Display descriptive statistics
st.subheader("Summary Statistics across both leagues: ")
st.dataframe(numeric_df.describe())

### --------------------------------------------- PAIRPLOT VISUALISATION ---------------------------------------------
df = numeric_df[['Goal_kicks_Attempted','Goals_Against', 'Launched_Passes_percent', 'Crosses_Faced', 'La_Liga']]
st.subheader("The English Premier League and La Liga have minimal visual differences between them.")
fig_pair = plt.figure(figsize=(12, 10))
pair_plot = sns.pairplot(df, hue='La_Liga', diag_kind='kde', palette='Set1', corner=False, plot_kws={'alpha': 0.6}, markers=["o","s"])
st.pyplot(pair_plot.figure)

st.subheader("We could say a close game is one where both teams are frequently threatening to score.")

### --------------------------------------------- REGRESSION MODELLING ---------------------------------------------
#Re-designing function fit for report displaying purposes, titles, hiding summaries etc
st.header("Can we use Polynomial Regression to pull out any relationships with how many goal kicks will be attempted by the opposing team?")
st.write("In hindsight, I should have split by team first and then compared the regression models for each league to see if there are any features that differ between leagues.")
plt.style.use("seaborn-v0_8-deep") # Use: print(plt.style.available) to check available styles

def lin_poly_Regression(predictor, response_var, ax):
    ax.scatter(predictor, response_var, s=10, alpha=0.3)
    ax.set_xlabel(predictor.name)
    ax.set_ylabel(response_var.name)
    x = pd.DataFrame({predictor.name : np.linspace(predictor.min(), predictor.max(), len(predictor)) })
    # Polynomial degree 1
    poly_1 = smf.ols(formula=f"{response_var.name} ~ 1 + {predictor.name}", data=numeric_df).fit()
    ax.plot(x, poly_1.predict(x), 'b-', label='Poly n=1 $R^2$=%.2f' % poly_1.rsquared,  alpha=0.9)
    # Polynomial degree 2
    poly_2 = smf.ols(formula=f"{response_var.name} ~ 1 + {predictor.name} + I({predictor.name} ** 2.0)", data=numeric_df).fit()
    ax.plot(x, poly_2.predict(x), 'g-', label='Poly n=2 $R^2$=%.2f' % poly_2.rsquared, alpha=0.9)
    # Polynomial degree 3
    poly_3 = smf.ols(formula=f"{response_var.name} ~ 1 + {predictor.name} + I({predictor.name} ** 2.0) + I({predictor.name} ** 3.0)", data=numeric_df).fit()
    ax.plot(x, poly_3.predict(x), 'r-', alpha=0.9,
            label='Poly n=3 $R^2$=%.2f' % poly_3.rsquared)
    ax.set_title(f"Does {predictor.name} Predict {response_var.name}")
    ax.legend(loc='lower right', frameon=True, fontsize='small')
    # Return the poly_3 model for summary display
    return poly_3

# Create the subplot figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)  # Made it taller for better display
predictors = [
    numeric_df['Goals_Against'],
    numeric_df['Launched_Passes_percent'],
    numeric_df['Crosses_Faced'], 
    numeric_df['Goal_Kicks_Average_Length'], 
    numeric_df['Goal_kicks_Launched_percent'],
    numeric_df['Passes_Completed_Over_40yrds']
]

axes = axes.flatten()
models = []  # Store models for summary display
for ax, pred in zip(axes, predictors):
    model = lin_poly_Regression(pred, numeric_df['Goal_kicks_Attempted'], ax)
    models.append(model)
plt.tight_layout()  # stops covering of axis labels
# Display the plot in Streamlit
st.pyplot(fig)
#Display model summaries in expandable sections
st.subheader("Model Summaries")
predictor_names = [pred.name for pred in predictors]
for i, (model, pred_name) in enumerate(zip(models, predictor_names)):
    with st.expander(f"Polynomial Degree 3 Summary: {pred_name}"):
        st.text(str(model.summary()))

### --------------------------------------------- RESIDUALS OF MODELLING ---------------------------------------------
# Streamlit version of residuals analysis

def lin_poly_Residuals(predictor, response_var, ax):
    ax.scatter(predictor, response_var, s=10, alpha=0.3, label='distribution actual data')
    ax.set_xlabel(predictor.name)
    ax.set_ylabel("Residuals of " + response_var.name)   
    # Fit the cubic polynomial regression
    poly_3 = smf.ols(formula=f"{response_var.name} ~ 1 + {predictor.name} + I({predictor.name} ** 2.0) + I({predictor.name} ** 3.0)", data=numeric_df).fit()
    residuals = poly_3.resid
    ax.scatter(predictor, residuals, color='red', s=15, alpha=0.7, label='Residuals')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)  # horizontal line at zero residual
    ax.set_title(f"Residuals of {response_var.name} vs {predictor.name}")
    ax.legend(loc='upper right', frameon=True, fontsize='small')
    return poly_3  # Return model for additional analysis if needed

# Streamlit display
st.header("Is Regression a trustworthy model in these cases?")
st.write("Examining residuals from cubic polynomial regression models")

# Create the subplot figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)  # Made taller for better display
predictors = [
    numeric_df['Goals_Against'],
    numeric_df['Launched_Passes_percent'],
    numeric_df['Crosses_Faced'], 
    numeric_df['Goal_Kicks_Average_Length'], 
    numeric_df['Goal_kicks_Launched_percent'],
    numeric_df['Passes_Completed_Over_40yrds']
]
axes = axes.flatten()
for ax, pred in zip(axes, predictors):
    lin_poly_Residuals(pred, numeric_df['Goal_kicks_Attempted'], ax)
plt.tight_layout()
# Display the plot in Streamlit
st.pyplot(fig)
st.write("Assumptions of the data: Linearity, Normality, Independence, Homoscedasticity (consistant variance).")
st.write("The top 3 residual plots are generally scattered around 0 with no visible trends, indicating a good model fit.")
st.write("In contrast, the bottom 3 residual plots show greater variance and distinct patterns emerging, suggesting a poor fit.")
st.write("This implies that the regression models corresponding to the top plots are more reliable.")

### --------------------------------------------- LOGISTIC REGRESSION MODELLING ---------------------------------------------
# Complete Streamlit ML Analysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

### --------------------------------------------- League Balance ---------------------------------------------
st.header("Can we predict what league they're in with the features that do show league specific patterns?")
st.write("A balanced dataset out the gate! This means we don't have to ues SMOTE")
balance_counts = numeric_df['La_Liga'].value_counts()
# Plotting the class balance
fig, ax = plt.subplots()
balance_counts.plot(kind='bar', ax=ax, color=['lightcoral','skyblue']) #keeping colouring consistent to avoid confusion
ax.set_xlabel("Class Labels")  # x-axis title
ax.set_ylabel("Number of Samples")  # y-axis title
ax.set_title("Class Distribution in Dataset")  # optional chart title
ax.set_xticklabels(['English Premier League', 'La Liga'], rotation=0)  # Set x-tick labels
st.pyplot(fig)

### --------------------------------------------- DATA PREP ---------------------------------------------
#df = numeric_df[['Goal_kicks_Attempted','Goals_Against', 'Passes_Completed_Over_40yrds', 
#                'Crosses_Faced', 'Goal_kicks_Launched_percent','Launched_Passes_percent','La_Liga']]
df = numeric_df[['Goal_kicks_Attempted','Goals_Against', 'Launched_Passes_percent', 'Crosses_Faced', 'La_Liga']]


### --------------------------------------------- NAIVE BAYES CLASSIFICATION ---------------------------------------------
X = df[['Goals_Against', 'Crosses_Faced','Launched_Passes_percent']]
y = df.La_Liga
st.subheader("Naive Bayes Classification")
nb = GaussianNB()
### --------------------------------------------- Train-Test Split Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5, stratify=y)
nb.fit(X_train, y_train) #training
st.write("**How well the trained classifier performs on the test data:**")
test_report = classification_report(y_test, nb.predict(X_test), output_dict=True)
test_df = pd.DataFrame(test_report).transpose()
st.dataframe(test_df)
st.write("The initial model is good at at distinguishing between La Liga and English Premier league!")

st.write("**How well the trained model performs on the entire dataset:**")
full_report = classification_report(y, nb.predict(X), output_dict=True)
full_df = pd.DataFrame(full_report).transpose()
st.dataframe(full_df)
st.write("f1 score drops, this indicates to me the training data is overfitted (performs better on the trained data than real observations)")
st.write("We can try using kFold-CrossValidation for a more robust/reliable evaluation -->")
### --------------------------------------------- kF-CV split analysis
st.write("**K-FOLD-CROSS-VALIDATION Performance:**")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
nb = GaussianNB()
cv_results = cross_validate(
    estimator=nb,
    X=X,
    y=y,
    cv=cv,
    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
    return_train_score=True)

cv_metrics = []
for metric in ['train_accuracy', 'test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']:
    scores = cv_results[metric]
    cv_metrics.append({
        'Metric': metric,
        'Mean': f"{scores.mean():.4f}",
        'Std': f"{scores.std():.4f}"
    })
cv_df = pd.DataFrame(cv_metrics)
st.dataframe(cv_df)

st.write("Training f1 score decreases: 0.7 down to 0.59 for k-Fold.")
st.write(" This indicates the single split was really overfitting. Real world performance is probably closer to 60%")
st.write("It also implies the Goalkeeper features used strongly distinguish between leagues")

### --------------------------------------------- MODEL CALIBRATION ANALYSIS ---------------------------------------------
st.header("Just by using the right model variables we can rein in our Calibration:")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Uncalibrated model: All 6 (incl. dodgy residual plots)")
    df = numeric_df[['Goal_kicks_Attempted','Goals_Against', 'Passes_Completed_Over_40yrds', 'Crosses_Faced', 'Goal_kicks_Launched_percent','Launched_Passes_percent','La_Liga']]
    X = df[['Goals_Against', 'Passes_Completed_Over_40yrds', 'Crosses_Faced', 'Goal_kicks_Launched_percent','Launched_Passes_percent']]
    nb.fit(X, y)
    X_probaR = nb.predict_proba(X)[:, 1]

    # Calibration curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(y, X_probaR, n_bins=9)
    fig_cal_uncal, ax_cal_uncal = plt.subplots(figsize=(6, 4))
    ax_cal_uncal.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated')
    ax_cal_uncal.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    ax_cal_uncal.set_xlabel('Mean Predicted Probability')
    ax_cal_uncal.set_ylabel('Fraction of Positives (Real Outcomes)')
    ax_cal_uncal.set_title('Calibration Curve')
    ax_cal_uncal.legend()
    ax_cal_uncal.grid()
    st.pyplot(fig_cal_uncal)

with col2:
    st.subheader("Uncalibrated model: Top 3 (strong residual plots)")
    X = df[['Goals_Against', 'Crosses_Faced','Launched_Passes_percent']]
    nb.fit(X, y)
    X_proba = nb.predict_proba(X)[:, 1]

    # Calibration curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(y, X_proba, n_bins=9)
    fig_cal_uncal, ax_cal_uncal = plt.subplots(figsize=(6, 4))
    ax_cal_uncal.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated')
    ax_cal_uncal.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    ax_cal_uncal.set_xlabel('Mean Predicted Probability')
    ax_cal_uncal.set_ylabel('Fraction of Positives (Real Outcomes)')
    ax_cal_uncal.set_title('Calibration Curve')
    ax_cal_uncal.legend()
    ax_cal_uncal.grid()
    st.pyplot(fig_cal_uncal)


### --------------------------------------------- Probability Distribution (Uncalibrated)
fig_prob_uncal, ax_prob_uncal = plt.subplots(figsize=(8, 5))
pd.DataFrame(X_proba).hist(bins=20, ax=ax_prob_uncal)
ax_prob_uncal.set_title("NB kF-CV Distribution of Probabilities (Uncalibrated, good variables)")
ax_prob_uncal.set_xlabel("Probability")
ax_prob_uncal.set_ylabel("Frequency")
st.pyplot(fig_prob_uncal)


### --------------------------------------------- Platt Scaling Calibration
st.header("Then we can use Platt Scaling to really hone in on a strong model:")

# Log Odds
log_odds = nb.predict_log_proba(X)[:, 1] - nb.predict_log_proba(X)[:, 0]

platt = LogisticRegression(solver='lbfgs')
platt.fit(log_odds.reshape(-1, 1), y)
calibrated_proba = platt.predict_proba(log_odds.reshape(-1, 1))[:, 1]

### --------------------------------------------- Calibrated curve
prob_true_cal, prob_pred_cal = calibration_curve(y, calibrated_proba, n_bins=8)

fig_cal_cal, ax_cal_cal = plt.subplots(figsize=(8, 6))
ax_cal_cal.plot(prob_pred_cal, prob_true_cal, marker='o', label='Calibrated')
ax_cal_cal.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
ax_cal_cal.set_xlabel('Mean Predicted Probability')
ax_cal_cal.set_ylabel('Fraction of Positives')
ax_cal_cal.set_title('Calibration Curve: Calibrated NB (Platt Scaling) on Actual Dataset')
ax_cal_cal.legend()
ax_cal_cal.grid()
st.pyplot(fig_cal_cal)

### --------------------------------------------- Probability Distribution (Calibrated)
fig_prob_cal, ax_prob_cal = plt.subplots(figsize=(8, 5))
pd.DataFrame(calibrated_proba).hist(bins=20, ax=ax_prob_cal)
ax_prob_cal.set_title("NB kF-CV Distribution of Probabilities With Platt Scaler")
ax_prob_cal.set_xlabel("Probability")
ax_prob_cal.set_ylabel("Frequency")
st.pyplot(fig_prob_cal)

# Final Classification Report
calibrated_predictions = (platt.predict_proba(
    (nb.predict_log_proba(X)[:, 1] - nb.predict_log_proba(X)[:, 0]).reshape(-1, 1)
)[:, 1] >= 0.5).astype(int)

st.write("**Final Model Performance:**")
final_report = classification_report(y, calibrated_predictions, output_dict=True)
final_df = pd.DataFrame(final_report).transpose()
st.dataframe(final_df)

# Brier Score Comparison
st.subheader("Model Calibration Quality (Brier Score)")
df = numeric_df[['Goal_kicks_Attempted','Goals_Against', 'Passes_Completed_Over_40yrds', 'Crosses_Faced', 'Goal_kicks_Launched_percent','Launched_Passes_percent','La_Liga']]
X = df[['Goals_Against', 'Passes_Completed_Over_40yrds', 'Crosses_Faced', 'Goal_kicks_Launched_percent','Launched_Passes_percent']]
nb.fit(X, y)
X_proba = nb.predict_proba(X)[:, 1]

proba_uncal = nb.predict_proba(X)[:,1]
proba_calib = calibrated_proba

brier_uncal = brier_score_loss(y, proba_uncal)
brier_calib = brier_score_loss(y, proba_calib)

col1, col2 = st.columns(2)
with col1:
    st.metric("Uncalibrated Score: ", f"{brier_uncal:.4f}")
with col2:
    st.metric("Calibrated Score: ", f"{brier_calib:.4f}")

st.write("**Lower Brier scores indicate better calibration. A Brier score of 0 is perfect.**")

# Summary
st.subheader("Analysis Summary:")
st.write("The intial f1 csore of 0.7 on a single split means the model found distinguishable patterns between La Liga and The English Premier League.")
st.write("The drop when using kF-CV suggests the patterns aren't across Leagues but instead teams. Or the dataset isn't large enough.")
st.header("Based on keeper data, one league isn't more intense but different teams are.")
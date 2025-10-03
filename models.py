import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load dataset
df = pd.read_csv("diving_synthetic_profiles.csv")

# Logistic regression (Logit)
logit_model = smf.logit(
    "Incident ~ MaxDepth_m + BottomTime_min + AscentRate_m_per_min + SafetyStop + NDL_Violation",
    data=df
).fit()

print(logit_model.summary())

# Probit regression
probit_model = smf.probit(
    "Incident ~ MaxDepth_m + BottomTime_min + AscentRate_m_per_min + SafetyStop + NDL_Violation",
    data=df
).fit()

# Save output as HTML file
with open("regression_results.html", "w") as f:
    f.write(logit_model.summary().as_html())
    f.write("<br><br>")
    f.write(probit_model.summary().as_html())


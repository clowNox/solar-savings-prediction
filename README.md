# Solar Energy Consumption & Savings Analysis

## Project Overview
This project analyzes **solar energy consumption & cost savings** for households using **machine learning models**. It evaluates how **household size, solar generation, government subsidies, and setup costs** impact electricity bills, providing insights and predicting potential savings.

## Dataset Information
The dataset includes **solar energy consumption, government subsidies, and cost savings data** with **10,000+ rows**.

### Key Features
- **Household Info:** `Household_Size`, `House_Area_sqft`
- **Energy Usage:** `Monthly_Consumption_kWh`, `Solar_Generation_kWh`, `Energy_Sent_to_Grid_kWh`
- **Billing & Cost:** `Final_Bill_Before_Savings`, `Final_Bill_After_Savings`, `Subsidy_Amount`, `Solar_Setup_Cost`
- **Government Policies:** `Govt_Solar_Subsidy_%`, `Net_Metering_Credit_per_kWh`

## Data Analysis & Visualizations
The project includes **4 visualizations**:
1. **Solar Generation vs. Monthly Consumption** (Scatter Plot)  
2. **Impact of Household Size on Savings** (Box Plot)  
3. **Break-Even Years vs. Solar Setup Cost** (Scatter Plot)  
4. **Effect of Government Subsidies on Savings** (Scatter Plot)  

## Machine Learning Models
Trained models to predict **Final Electricity Bill After Savings (₹):**
- **Random Forest Regressor** (Ensemble model for accurate predictions)
- **Decision Tree Regressor** (Visualizes decision-making steps)
- **Linear Regression** (Estimates relationships between features)
- **XGBoost Regressor** (Uses gradient boosting for accuracy)

## Decision Tree Visualization
The **Decision Tree Regressor** predicts **solar energy cost savings**, showing:
- **Feature importance** (e.g., `"Solar_Setup_Cost"` as the root node).
- **Decision paths** for predicting savings.
- **Final predicted values** in leaf nodes.


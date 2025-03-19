# ☀️ Solar Savings Prediction Project

## 📌 Project Overview
This project predicts **household electricity savings** using **solar panels**. It leverages **machine learning models** to estimate the **break-even period**, evaluate the impact of **government subsidies**, and forecast long-term savings.

## 🚀 Features
- **Data Analysis**: Household electricity consumption, solar generation, and financial savings.
- **4 Machine Learning Models**: Random Forest, Linear Regression, Decision Tree, and SVR.
- **Break-even Analysis**: How long does it take for solar panels to pay off?
- **Government Incentives**: Effect of subsidies and net metering on savings.
- **Predictive Model**: Estimate future solar savings for new households.

## 📂 Project Structure
📁 **solar_savings_dataset_updated.csv** → Dataset used for analysis.  
📁 **solar_savings_rf_model.pkl** → Trained model (Random Forest).  
📁 **Solar_Savings_Notebook.ipynb** → Jupyter Notebook with **code, visualizations, and ML models**.  
📁 **Solar_Savings_Documentation.md** → Detailed **project report**.

## 🏆 Best Model Performance
| Model                         | MAE   | RMSE  | R² Score |
|--------------------------------|-------|-------|---------|
| **Random Forest Regressor**    | 62.95 | 96.79 | 0.9986  ✅ |
| **Linear Regression**          | 0.00  | 0.00  | **1.0000** 🚨 Overfitting |
| **Decision Tree Regressor**    | 130.98| 184.38| 0.9950  |
| **Support Vector Regressor**   | 1813.06 | 2267.35 | 0.2396 ❌ |

## 📌 Conclusion
✅ **Solar energy adoption leads to massive electricity savings!**  
✅ **The Random Forest model provides the best predictions.**  
✅ **Government incentives play a crucial role in financial savings.**  

🔹 **Future Enhancements:**
- **Integrating real-world electricity billing datasets.**
- **Building an interactive web dashboard for solar savings predictions.**
- **Exploring deep learning for improved accuracy.**

🌍 **Go Solar, Save Money, and Reduce Carbon Emissions!** 🚀


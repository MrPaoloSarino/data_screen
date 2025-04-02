# 📊 Screen Time Analysis & Prediction Dashboard

This project presents an interactive web dashboard built with Streamlit to analyze and visualize children's screen time habits based on aggregated survey data. It explores trends across age, gender, and day type (weekday/weekend) and includes a simple predictive component to forecast average total screen time trends.

## ✨ Features

*   **Interactive Filtering:** Filter data by Age Range, Gender, and Day Type using sidebar controls.
*   **Key Metrics:** View overall average Total Screen Time, Recreational Proportion, and Educational Proportion based on filtered data.
*   **Sample Size Overview:** Visualize the number of participants for each age group within the filtered data.
*   **Descriptive Visualizations:**
    *   Line charts showing trends in screen time hours (Total, Educational, Recreational) by age.
    *   Line charts showing trends in screen time *proportions* (Educational vs. Recreational) by age.
    *   Bar charts comparing average screen time between Weekdays and Weekends.
    *   Bar chart showing the calculated *increase* in screen time on weekends vs. weekdays by age.
    *   Grouped bar charts comparing screen time across Genders and Day Types.
    *   Box plots showing the distribution of average screen time by Age Group.
    *   Scatter plot exploring the relationship between Educational and Recreational hours.
    *   Heatmap visualizing average total screen time across Age Groups and Genders.
    *   Radar chart comparing Educational, Recreational, and Total hours profiles across Genders.
*   **🔮 Predictive Analytics:**
    *   Fits a Polynomial Regression model to the average total screen time trend based on filtered data.
    *   Predicts (extrapolates) the average total screen time trend for a few years beyond the observed data range (configurable).
    *   Visualizes the fitted curve and the predicted trend line.
    *   Displays the R² score for the model fit on observed data.
    *   Provides clear caveats about the nature of the prediction.
*   **Data View:** Display the filtered data in a table format.

## 🛠️ Technology Stack

*   **Python 3.x**
*   **Streamlit:** For creating the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **Plotly:** For creating interactive visualizations.
*   **Scikit-learn:** For implementing the Polynomial Regression model.

## 💾 Data

This dashboard relies on a pre-processed CSV file:

*   `screen_time_cleaned_analyzed.csv`

This file should contain aggregated screen time data with columns like `Age`, `Gender`, `Day_Type`, `Total_Hours`, `Educational_Hours`, `Recreational_Hours`, `Sample_Size`, `Age_Group`, etc.

**-> IMPORTANT:** This CSV file **must** be placed in the **same directory** as the Python script (`screen_time_dashboard.py`) for the dashboard to load the data correctly.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install streamlit pandas plotly scikit-learn
    ```
    *(Alternatively, if you create a `requirements.txt` file, you can use `pip install -r requirements.txt`)*
3.  **Place the data file:** Ensure `screen_time_cleaned_analyzed.csv` is located in the root directory of the cloned repository (alongside the `.py` script).

## 🚀 Usage

Navigate to the project directory in your terminal and run the Streamlit application:

```bash
streamlit run screen_time_dashboard.py

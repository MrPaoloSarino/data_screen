import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score # To show model fit

# --- Page Configuration ---
st.set_page_config(
    page_title="Screen Time Analysis Dashboard",
    page_icon="🔮", # Changed icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Function for Insights ---
def render_insight(text, level="info"):
    """Adds a styled insight box below a chart."""
    icon = {"info": "💡", "warning": "⚠️", "success": "✅"}.get(level, "💡")
    st.markdown(f"> {icon} **Insight:** {text}")
    st.write("") # Add a little space

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    # (Keep the existing load_data function as is)
    if not os.path.exists(file_path):
        st.error(f"Error: File not found at {file_path}. Make sure 'screen_time_cleaned_analyzed.csv' is in the same directory as the script.")
        return None
    try:
        df = pd.read_csv(file_path)
        df['Age'] = df['Age'].astype(int)
        df['Sample_Size'] = df['Sample_Size'].astype(int)
        if 'Age_Group' in df.columns:
             df['Age_Group'] = pd.Categorical(df['Age_Group'], categories=['5-7', '8-11', '12-15'], ordered=True)
        # Calculate proportions if they weren't saved correctly
        if 'Total_Hours' in df.columns and df['Total_Hours'].notna().all() and df['Total_Hours'].gt(0).all():
             if 'Educational_Proportion' not in df.columns and 'Educational_Hours' in df.columns:
                 df['Educational_Proportion'] = (df['Educational_Hours'] / df['Total_Hours']).fillna(0)
             if 'Recreational_Proportion' not in df.columns and 'Recreational_Hours' in df.columns:
                df['Recreational_Proportion'] = (df['Recreational_Hours'] / df['Total_Hours']).fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# --- Predictive Modeling Function ---
def train_and_predict(df_agg, degree=2, predict_up_to_age=18):
    """Trains a polynomial regression model and predicts future values."""
    if df_agg.empty or len(df_agg) < degree + 1:
        return None, None, None, None # Not enough data to train

    X = df_agg[['Age']]
    y = df_agg['Total_Hours']

    try:
        # Create and train the polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)

        # Predict on existing data to get the fitted line
        y_pred_fitted = model.predict(X)
        r2 = r2_score(y, y_pred_fitted) # Calculate R-squared

        # Prepare ages for future prediction
        min_age_pred = df_agg['Age'].max() + 1
        if predict_up_to_age < min_age_pred:
             predict_up_to_age = min_age_pred # Ensure we predict at least one step ahead

        X_pred = pd.DataFrame({'Age': np.arange(min_age_pred, predict_up_to_age + 1)})

        # Predict future values
        y_pred_future = model.predict(X_pred)

        # Combine existing ages and future ages for plotting
        plot_X = np.arange(df_agg['Age'].min(), predict_up_to_age + 1).reshape(-1, 1)
        plot_y = model.predict(plot_X)

        return plot_X.flatten(), plot_y, X_pred['Age'].values, y_pred_future, r2
    except Exception as e:
        st.error(f"Error during model training/prediction: {e}")
        return None, None, None, None, None


# --- Load Data ---
DATA_FILE = 'screen_time_cleaned_analyzed.csv'
df = load_data(DATA_FILE)

# --- Main Application ---
if df is not None:
    st.title("🔮 Screen Time Analysis & Prediction Dashboard") # Updated Title
    st.markdown("""
    Explore average screen time trends and view simple trend predictions.
    Use the filters and prediction settings in the sidebar. Hover over charts for details.
    *Data Source: Aggregated survey data. Predictions are based on observed trends and assume these trends continue.*
    """)

    # --- Sidebar ---
    st.sidebar.header("Filters")
    # (Keep existing filters: Age Range, Gender, Day Type)
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    selected_age_range = st.sidebar.slider(
        "Select Age Range (for visualization)", min_age, max_age, (min_age, max_age) # Clarified purpose
    )

    gender_options = sorted(df['Gender'].unique())
    selected_genders = st.sidebar.multiselect(
        "Select Gender(s)", gender_options, default=gender_options
    )

    day_type_options = sorted(df['Day_Type'].unique())
    selected_day_types = st.sidebar.multiselect(
        "Select Day Type(s)", day_type_options, default=day_type_options
    )

    st.sidebar.header("Prediction Settings")
    poly_degree = st.sidebar.slider(
        "Polynomial Degree for Trend Fit", min_value=1, max_value=5, value=2,
        help="Degree=1 is a straight line. Higher degrees fit curves but risk overfitting."
    )
    predict_years_future = st.sidebar.slider(
        "Predict Trend Up To Age", min_value=max_age + 1, max_value=max_age + 5, value=max_age + 3, # Predict 3 years ahead default
        help="Select how many years beyond the maximum observed age (15) to forecast the trend."
        )

    # --- Filter Data for Visualization ---
    # Filter based on sidebar selections for display purposes
    viz_filtered_df = df[
        (df['Age'] >= selected_age_range[0]) &
        (df['Age'] <= selected_age_range[1]) &
        (df['Gender'].isin(selected_genders if selected_genders else gender_options)) &
        (df['Day_Type'].isin(selected_day_types if selected_day_types else day_type_options))
    ].copy()

    # --- Prepare Data for Modeling (Aggregate based on filters) ---
    # Group the filtered data by age and calculate the mean total hours for modeling
    model_df_agg = viz_filtered_df.groupby('Age', observed=False)['Total_Hours'].mean().reset_index()


    if viz_filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # --- Dashboard Content ---

        # (Keep Row 1: Key Metrics & Sample Overview)
        st.header("Key Metrics & Sample Overview (Filtered)")
        # ... (code for KPI cards and Sample Size chart remains the same) ...
        col1, col2, col3, col4 = st.columns(4)
        avg_total_hours = viz_filtered_df['Total_Hours'].mean()
        avg_rec_prop = viz_filtered_df['Recreational_Proportion'].mean() * 100
        avg_edu_prop = viz_filtered_df['Educational_Proportion'].mean() * 100
        total_sample_represented = viz_filtered_df.groupby('Age')['Sample_Size'].first().sum()
        with col1: st.metric(label="Avg. Total (Hours/Day)", value=f"{avg_total_hours:.2f}")
        with col2: st.metric(label="Avg. Rec. Prop (%)", value=f"{avg_rec_prop:.1f}%")
        with col3: st.metric(label="Avg. Edu. Prop (%)", value=f"{avg_edu_prop:.1f}%")
        with col4: st.metric(label="Unique Participants (Filtered)", value=f"{total_sample_represented:,}")

        st.subheader("Sample Size by Age (Filtered Data)")
        sample_df = viz_filtered_df.groupby('Age', observed=False)['Sample_Size'].first().reset_index()
        if not sample_df.empty:
            fig_sample = px.bar(sample_df, x='Age', y='Sample_Size', title="Sample Size per Age Group in Filtered Data", labels={'Sample_Size': 'Number of Participants'}, text_auto=True)
            st.plotly_chart(fig_sample, use_container_width=True)
            render_insight("This shows the number of participants surveyed for each age included in the current filtered view.")


        st.markdown("---")

        # --- Prediction Section ---
        st.header("Predictive Trend Analysis")
        st.markdown(f"Fitting a Polynomial Regression (Degree={poly_degree}) to the average **Total Hours** based on the filtered data and predicting up to age **{predict_years_future}**.")

        # Train model and get predictions
        plot_ages, plot_trend_line, future_ages, future_predictions, r_squared = train_and_predict(
            model_df_agg, degree=poly_degree, predict_up_to_age=predict_years_future
        )

        if plot_ages is not None:
            # Create the plot
            fig_pred = go.Figure()

            # 1. Scatter plot of the actual aggregated data points used for modeling
            fig_pred.add_trace(go.Scatter(
                x=model_df_agg['Age'],
                y=model_df_agg['Total_Hours'],
                mode='markers',
                name='Filtered Average Total Hours',
                marker=dict(color='blue', size=8)
            ))

            # 2. Plot the full trend line (fitted + predicted)
            fig_pred.add_trace(go.Scatter(
                x=plot_ages,
                y=plot_trend_line,
                mode='lines',
                name=f'Polynomial Fit (Degree {poly_degree})',
                line=dict(color='red', width=2)
            ))

            # 3. Highlight the predicted points (optional, can make plot busy)
            # fig_pred.add_trace(go.Scatter(
            #     x=future_ages,
            #     y=future_predictions,
            #     mode='markers',
            #     name='Predicted Values',
            #     marker=dict(color='red', symbol='x', size=10)
            # ))

            # Add vertical line separating observed from predicted
            max_observed_age = model_df_agg['Age'].max()
            fig_pred.add_vline(x=max_observed_age + 0.5, line_width=2, line_dash="dash", line_color="grey",
                              annotation_text="Prediction Start", annotation_position="top left")

            fig_pred.update_layout(
                title=f"Average Total Screen Time Trend and Prediction (R² = {r_squared:.3f})",
                xaxis_title="Age",
                yaxis_title="Average Total Hours per Day",
                legend_title_text='Data/Model'
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Display predicted values
            st.subheader("Predicted Average Total Hours")
            pred_df = pd.DataFrame({'Age': future_ages, 'Predicted Avg Total Hours': future_predictions.round(2)})
            st.dataframe(pred_df)

            render_insight(f"The red line shows the trend fitted to the filtered average data (blue points) using Polynomial Regression (Degree {poly_degree}). The dashed grey line marks the end of observed data; values beyond this are extrapolated predictions. The R² value ({r_squared:.3f}) indicates how well the model fits the *observed* average trend (1 is perfect fit).")
            render_insight("Predictions assume the observed trend continues exactly as modeled. Real-world screen time may change differently due to factors not included in this simple age-based model (e.g., life events, technology changes, policy interventions). Use predictions cautiously.", level="warning")

        else:
            st.warning(f"Could not train predictive model. Need at least {poly_degree + 1} distinct age points in the filtered data.")


        st.markdown("---")

        # --- Existing Sections (Core Trends, Comparisons, Advanced Viz) ---
        st.header("Descriptive Analysis")
        st.markdown("*(These charts show patterns within the observed data based on filters)*")

        # Row: Core Trends
        # ... (Keep code for Hours Trend and Proportion Trend charts + insights) ...
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Screen Time Hours Trend by Age")
            # ... (existing code) ...
            trend_df = model_df_agg # Use the already aggregated df
            trend_df_melt = trend_df.melt(id_vars=['Age'], value_vars=['Total_Hours'], var_name='Screen Time Type', value_name='Average Hours') # Simplified for this section now
            # Add Edu/Rec lines if desired, requires different aggregation
            # trend_df_detail = viz_filtered_df.groupby('Age', observed=False)[['Educational_Hours', 'Recreational_Hours', 'Total_Hours']].mean().reset_index()
            # trend_df_melt = trend_df_detail.melt(id_vars=['Age'], value_vars=['Educational_Hours', 'Recreational_Hours', 'Total_Hours'], var_name='Screen Time Type', value_name='Average Hours')

            if not trend_df_melt.empty:
                 fig_trend = px.line(trend_df_melt, x='Age', y='Average Hours', color='Screen Time Type', title="Average Screen Time Hours vs. Age", labels={'Average Hours': 'Avg. Hours per Day'}, markers=True)
                 st.plotly_chart(fig_trend, use_container_width=True)
                 render_insight("Tracks the average daily screen time as age increases within the filtered data.")
            else: st.info("No data for Hours Trend chart.")


        with col_chart2:
            st.subheader("Screen Time Proportion Trend by Age")
            # ... (existing code) ...
            if 'Educational_Proportion' in viz_filtered_df.columns and 'Recreational_Proportion' in viz_filtered_df.columns and viz_filtered_df['Educational_Proportion'].notna().all() and viz_filtered_df['Recreational_Proportion'].notna().all():
                prop_trend_df = viz_filtered_df.groupby('Age', observed=False)[['Educational_Proportion', 'Recreational_Proportion']].mean().reset_index()
                prop_trend_melt = prop_trend_df.melt(id_vars=['Age'], value_vars=['Educational_Proportion', 'Recreational_Proportion'], var_name='Screen Time Type', value_name='Proportion')
                prop_trend_melt['Screen Time Type'] = prop_trend_melt['Screen Time Type'].str.replace('_Proportion', '')
                if not prop_trend_melt.empty:
                    fig_prop_trend = px.line(prop_trend_melt, x='Age', y='Proportion', color='Screen Time Type', title="Proportion of Screen Time vs. Age", labels={'Proportion': 'Avg. Proportion of Total Time'}, markers=True)
                    fig_prop_trend.update_layout(yaxis_tickformat=".0%")
                    st.plotly_chart(fig_prop_trend, use_container_width=True)
                    render_insight("Shows the percentage split between Educational vs. Recreational time across ages.")
                else: st.info("No data for Proportion Trend chart.")
            else: st.warning("Proportion columns needed or contain invalid data.")


        # Row: Comparisons
        # ... (Keep code for Weekend Increase and Gender Comparison charts + insights) ...
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
             st.subheader("Weekday vs. Weekend Increase (Total Hours)")
             # ... (existing code) ...
             try:
                pivot_day_df = viz_filtered_df.pivot_table(index=['Age', 'Gender'], columns='Day_Type', values='Total_Hours', observed=False)
                if 'Weekday' in pivot_day_df.columns and 'Weekend' in pivot_day_df.columns:
                    pivot_day_df['Weekend_Increase'] = pivot_day_df['Weekend'] - pivot_day_df['Weekday']
                    agg_increase_df = pivot_day_df.reset_index().groupby('Age')['Weekend_Increase'].mean().reset_index()
                    if not agg_increase_df.empty:
                        fig_increase = px.bar(agg_increase_df, x='Age', y='Weekend_Increase', title="Average Increase in Total Screen Time on Weekends", labels={'Weekend_Increase': 'Avg. Additional Hours on Weekend vs. Weekday'})
                        st.plotly_chart(fig_increase, use_container_width=True)
                        render_insight("Quantifies the average jump in total screen time on weekends compared to weekdays.")
                    else: st.info("Could not calculate weekend increase.")
                else: st.info("Need both Weekday and Weekend data selected.")
             except Exception as e: st.warning(f"Could not generate Weekend Increase chart. Error: {e}")

        with col_comp2:
             st.subheader("Average Total Screen Time by Gender and Day Type")
             # ... (existing code) ...
             gender_comp_df = viz_filtered_df.groupby(['Gender', 'Day_Type'], observed=False)['Total_Hours'].mean().reset_index()
             if not gender_comp_df.empty:
                 fig_gender = px.bar(gender_comp_df, x='Gender', y='Total_Hours', color='Day_Type', title="Avg. Total Screen Time by Gender and Day Type", labels={'Total_Hours': 'Avg. Total Hours per Day'}, barmode='group')
                 st.plotly_chart(fig_gender, use_container_width=True)
                 render_insight("Compares total daily screen time across genders, split by day type.")
             else: st.info("No data for Gender Comparison chart.")

        # Row: Advanced Visualizations
        # ... (Keep code for Box Plot, Radar Chart, Scatter Plot, Heatmap + insights) ...
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            st.subheader("Distribution of Average Total Hours by Age Group")
            # ... (existing code) ...
            if 'Age_Group' in viz_filtered_df.columns and not viz_filtered_df.empty:
                 fig_box = px.box(viz_filtered_df, x='Age_Group', y='Total_Hours', color='Day_Type', title="Distribution of Avg. Total Screen Time by Age Group", labels={'Total_Hours': 'Avg. Total Hours per Day'}, points="all")
                 st.plotly_chart(fig_box, use_container_width=True)
                 render_insight("Shows the spread of average total screen time values within each age group.")
            else: st.info("Box plot requires 'Age_Group' column.")

            st.subheader("Screen Time Profile by Gender (Average Hours)")
            # ... (existing code) ...
            radar_df = viz_filtered_df.groupby('Gender')[['Educational_Hours', 'Recreational_Hours', 'Total_Hours']].mean().reset_index()
            if not radar_df.empty and len(radar_df) > 0 and all(c in radar_df.columns for c in ['Educational_Hours', 'Recreational_Hours', 'Total_Hours']):
                 metrics = ['Educational_Hours', 'Recreational_Hours', 'Total_Hours']
                 metric_labels = ['Educational', 'Recreational', 'Total']
                 fig_radar = go.Figure()
                 max_val = viz_filtered_df[metrics].max().max()
                 for index, row in radar_df.iterrows():
                     values = row[metrics].values.flatten().tolist(); values += values[:1]
                     categories = metric_labels + metric_labels[:1]
                     fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=row['Gender']))
                 fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_val + 1])), showlegend=True, title="Comparison of Avg. Screen Time Types by Gender")
                 st.plotly_chart(fig_radar, use_container_width=True)
                 render_insight("Multi-dimensional comparison of screen time types across genders.")
            else: st.info("Radar chart requires data for selected genders and screen types.")

        with adv_col2:
            st.subheader("Educational vs. Recreational Hours")
            # ... (existing code) ...
            if not viz_filtered_df.empty and 'Educational_Hours' in viz_filtered_df.columns and 'Recreational_Hours' in viz_filtered_df.columns:
                 fig_scatter = px.scatter(viz_filtered_df, x='Educational_Hours', y='Recreational_Hours', color='Gender', size='Total_Hours', hover_data=['Age', 'Day_Type', 'Sample_Size'], title="Educational vs. Recreational Average Screen Time", labels={'Educational_Hours': 'Avg. Educational Hours'})
                 st.plotly_chart(fig_scatter, use_container_width=True)
                 render_insight("Explores the relationship between average educational and recreational hours.")
            else: st.info("Scatter plot requires Educational/Recreational hours data.")

            st.subheader("Average Total Hours: Age Group vs. Gender")
            # ... (existing code) ...
            if 'Age_Group' in viz_filtered_df.columns and not viz_filtered_df.empty:
                 try:
                     heatmap_df = viz_filtered_df.pivot_table(index='Age_Group', columns='Gender', values='Total_Hours', aggfunc='mean', observed=False)
                     fig_heatmap = px.imshow(heatmap_df, text_auto=".1f", aspect="auto", color_continuous_scale=px.colors.sequential.Plasma, title="Heatmap of Average Total Screen Time (Hours)")
                     st.plotly_chart(fig_heatmap, use_container_width=True)
                     render_insight("Grid view comparing average total screen time across age groups and genders.")
                 except Exception as e: st.warning(f"Could not generate heatmap. Error: {e}")
            else: st.info("Heatmap requires 'Age_Group' column.")


        # Row : Data Table (Keep as is)
        st.markdown("---")
        st.header("Filtered Data View")
        # ... (existing code) ...
        display_cols = [col for col in ['Age', 'Age_Group', 'Gender', 'Day_Type', 'Sample_Size', 'Educational_Hours', 'Recreational_Hours', 'Total_Hours', 'Educational_Proportion', 'Recreational_Proportion'] if col in viz_filtered_df.columns]
        st.dataframe(viz_filtered_df[display_cols].round(2))


else:
    st.warning("Data could not be loaded. Please ensure the CSV file exists and is readable.")


# --- Footer ---
st.markdown("---")
st.caption("Dashboard created using Streamlit, Plotly, and Scikit-learn.")
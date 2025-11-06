import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- NEW IMPORT ---
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    
# Set app configuration
st.set_page_config(layout="wide")
st.title("🍚 WFP Food Price Data Analysis")
st.markdown("---")

# Use st.cache_data for faster loading/processing.
@st.cache_data
def load_and_clean_data():
    """
    Loads, combines, cleans, and preprocesses the food price data.
    """
    try:
        # Load the two parts of the dataset created by the notebook's initial script
        df_part1 = pd.read_csv('wfpvam_foodprices_part1.csv', low_memory=False)
        df_part2 = pd.read_csv('wfpvam_foodprices_part2.csv', low_memory=False)

        # Combine both parts
        df_combined = pd.concat([df_part1, df_part2], ignore_index=True)

        # --- Data Cleaning and Feature Engineering ---
        if 'mp_commoditysource' in df_combined.columns:
            df_combined = df_combined.drop(columns=['mp_commoditysource'])
        if 'adm1_name' in df_combined.columns:
            df_combined['adm1_name'].fillna('Unknown', inplace=True)

        df_combined['mp_date'] = pd.to_datetime(
            df_combined['mp_year'].astype(str) + '-' + df_combined['mp_month'].astype(str) + '-01'
        )
        df_combined = df_combined.drop(columns=['mp_month', 'mp_year'])

        id_columns = [col for col in df_combined.columns if '_id' in col]
        for col in id_columns:
            df_combined[col] = df_combined[col].astype(str)

        df_combined['mp_price'] = pd.to_numeric(df_combined['mp_price'], errors='coerce')
        if df_combined['mp_price'].isnull().any():
            median_price = df_combined['mp_price'].median()
            df_combined['mp_price'].fillna(median_price, inplace=True)

        df_combined.drop_duplicates(inplace=True)

        lower_bound = df_combined['mp_price'].quantile(0.01)
        upper_bound = df_combined['mp_price'].quantile(0.99)
        df_cleaned = df_combined[(df_combined['mp_price'] >= lower_bound) & (df_combined['mp_price'] <= upper_bound)].copy()

        df_cleaned['month'] = df_cleaned['mp_date'].dt.month
        df_cleaned['year'] = df_cleaned['mp_date'].dt.year

        return df_cleaned

    except FileNotFoundError:
        st.error(
            "🛑 **File Not Found Error**\n\n"
            "Please ensure the required files: **'wfpvam_foodprices_part1.csv'** "
            "and **'wfpvam_foodprices_part2.csv'** are in the same directory."
        )
        return pd.DataFrame() 
    
# --- NEW FUNCTIONS FOR PROPHET ---
@st.cache_data
def prepare_prophet_data(df):
    """Aggregates data to Global Average Price time series (ds, y)."""
    prophet_df = df.groupby('mp_date')['mp_price'].mean().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

@st.cache_resource
def run_prophet_model(prophet_df):
    """Initializes, trains, and forecasts using the Prophet model."""
    # Note: Using try/except here helps catch potential Prophet initialization issues if the library is buggy.
    try:
        prophet_model = Prophet(
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        prophet_model.fit(prophet_df)

        # Create a future dataframe to forecast 1 year (12 months)
        future = prophet_model.make_future_dataframe(periods=12, freq='MS') 
        prophet_forecast = prophet_model.predict(future)
        return prophet_model, prophet_forecast
    except Exception as e:
        st.error(f"Prophet Model Failed to Initialize/Run: {e}")
        st.warning("This is often due to an incompatible 'pystan' dependency. Please re-run the installation steps provided above.")
        return None, None
# --- END NEW FUNCTIONS ---

# Load Data
df_cleaned = load_and_clean_data()

# Only display content if data loading was successful
if not df_cleaned.empty:
    st.sidebar.title("Configuration")
    st.sidebar.success(f"Data Loaded and Cleaned: {len(df_cleaned):,} records")

    # --- 1-4. Global Analysis Sections ---
    st.header("1. Cleaned and Preprocessed Data (Head)")
    st.dataframe(df_cleaned.head(10))
    st.markdown("---")

    st.header("2. Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df_cleaned['mp_price'], bins=50, kde=True, color='skyblue', ax=ax1)
    ax1.set_title('Distribution of Food Prices (After Outlier Removal)')
    st.pyplot(fig1)
    st.markdown("---")

    st.header("3. Global Average Price Trend Over Time")
    monthly_avg_price = df_cleaned.groupby('mp_date')['mp_price'].mean().resample('M').mean()
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    monthly_avg_price.plot(kind='line', marker='o', linestyle='-', markersize=4, color='orange', alpha=0.8, ax=ax2)
    ax2.set_title('Global Average Food Price Trend Over Time')
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)
    st.markdown("---")

    st.header("4. Monthly Price Seasonality (Heatmap)")
    monthly_avg_pivot = df_cleaned.pivot_table(values='mp_price', index='month', columns='year', aggfunc='mean')
    fig3, ax3 = plt.subplots(figsize=(15, 8))
    sns.heatmap(monthly_avg_pivot, cmap='YlGnBu', annot=True, fmt=".0f", linewidths=.5, cbar_kws={'label': 'Average Price'}, 
                yticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ax=ax3)
    ax3.set_title('Monthly Price Seasonality (Average Price by Month and Year)')
    st.pyplot(fig3)
    st.markdown("---")
    
    # --- 5. Interactive Country and Commodity Analysis (Cascading Filters) ---

    st.header("5. Interactive Country & Commodity Analysis")
    st.sidebar.header("Filter Data")

    country_list = sorted(df_cleaned['adm0_name'].unique())
    default_country_index = country_list.index('Afghanistan') if 'Afghanistan' in country_list else 0

    # 1. Country Selector
    selected_country = st.sidebar.selectbox(
        'Select Country:', 
        country_list, 
        index=default_country_index
    )
    
    # 2. Filter data for the selected country
    df_country_filtered = df_cleaned[df_cleaned['adm0_name'] == selected_country]
    
    # 3. Commodity list updates based on selected country
    commodity_list = sorted(df_country_filtered['cm_name'].unique())
    default_commodity_index = commodity_list.index('Wheat') if 'Wheat' in commodity_list else 0

    # 4. Commodity Selector
    selected_commodity = st.sidebar.selectbox(
        'Select Commodity:', 
        commodity_list, 
        index=default_commodity_index
    )

    # Final Filter
    df_filtered = df_country_filtered[
        (df_country_filtered['cm_name'] == selected_commodity)
    ]

    if df_filtered.empty:
        st.warning(f"No data found for {selected_commodity} in {selected_country}. Please select a different combination.")
    else:
        st.subheader(f"Price Analysis for: {selected_commodity} in {selected_country}")

        # Plot A: Specific Price Trend Over Time
        specific_monthly_price = df_filtered.groupby('mp_date')['mp_price'].mean()
        fig_specific, ax_specific = plt.subplots(figsize=(12, 6))
        specific_monthly_price.plot(kind='line', color='darkred', linewidth=2, ax=ax_specific)
        ax_specific.set_title(f'Average Price Trend: {selected_commodity} in {selected_country}')
        ax_specific.set_xlabel('Date')
        ax_specific.set_ylabel('Average Price')
        ax_specific.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig_specific)

        # Plot B: Price Distribution by Market Type (Box Plot)
        if 'mkt_name' in df_filtered.columns and len(df_filtered['mkt_name'].unique()) > 1:
            st.subheader(f"Price Distribution by Market")
            market_counts = df_filtered['mkt_name'].value_counts()
            top_markets = market_counts.nlargest(10).index
            df_top_markets = df_filtered[df_filtered['mkt_name'].isin(top_markets)]
            
            fig_market, ax_market = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='mkt_name', y='mp_price', data=df_top_markets, palette='viridis', ax=ax_market)
            ax_market.set_title(f'Price Distribution for {selected_commodity} by Market (Top {len(top_markets)})')
            ax_market.set_xlabel('Market Name')
            ax_market.set_ylabel('Price')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_market)
        else:
            st.info("Market distribution data is not meaningful for this filtered selection (fewer than 2 markets).")
    
    st.markdown("---")

    # --- 6. Future Price Prediction (Prophet Model) ---
    st.header("6. Future Price Prediction (Prophet Model)")

    if prophet_available:
        # Check for model success before plotting
        prophet_df = prepare_prophet_data(df_cleaned)
        prophet_model, prophet_forecast = run_prophet_model(prophet_df)

        if prophet_model is not None and prophet_forecast is not None:
            st.info("The prediction is based on the **Global Average Price** trend across all countries for a stable 12-month forecast.")
            
            # 1. Forecast Plot
            fig_forecast = prophet_model.plot(prophet_forecast)
            plt.title("Prophet Forecast: Historical Fit and 1-Year Future Prediction")
            st.pyplot(fig_forecast)

            # 2. Components Plot
            fig_components = prophet_model.plot_components(prophet_forecast)
            st.pyplot(fig_components)
    else:
        st.warning(
            "Prophet forecasting is skipped because the library is not installed. "
            "Please run `pip install prophet` in your terminal and restart the app to enable this feature."
        )
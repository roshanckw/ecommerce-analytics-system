
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }

    /* Subtle Light Background for the entire app */
    .stApp > header {
        background-color: transparent;
    }
    .stApp [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC; 
    }

    /* Gradient Main Title */
    h1 {
        font-weight: 700 !important;
        background: -webkit-linear-gradient(45deg, #1E40AF, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }

    /* Main Headers (h2) - Left Accent Dashboard Style */
    h2 {
        font-weight: 600 !important;
        color: #1E293B !important;
        padding-left: 12px;
        border-left: 5px solid #7C3AED;
        margin-top: 2rem !important;
        border-bottom: none !important;
        padding-bottom: 0px !important;
    }

    /* Subheaders (h3) - Fading Gradient Underline */
    h3 {
        font-weight: 500 !important;
        color: #475569 !important;
        border-bottom: 2px solid;
        border-image: linear-gradient(to right, #7C3AED 0%, rgba(124, 58, 237, 0) 100%) 1;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem !important;
    }

    /* Prettier File Uploader */
    [data-testid="stFileUploadDropzone"] {
        border-radius: 12px;
        border: 2px dashed #CBD5E1;
        background-color: #FFFFFF;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #7C3AED;
        background-color: #F3F0FF;
    }

    /* Alerts / Warning / Info Boxes */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border: none !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* Polished Buttons */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
        border: 1px solid #E2E8F0;
        background-color: white;
        font-weight: 500;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: #7C3AED;
        color: #7C3AED;
    }
    
    /* Soften DataFrames & Cards */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #F1F5F9;
        background-color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Intelligent E-Commerce data Analytics System ")
st.header("Dataset Upload")

uploaded_file = st.file_uploader(
    "Upload an e-commerce CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    
    

    #  Detect new file
    current_file = uploaded_file.name

    if "last_uploaded_file" not in st.session_state:
        st.session_state["last_uploaded_file"] = None

    

   

    if current_file != st.session_state["last_uploaded_file"]:
        # Clear all keys except last_uploaded_file tracker
        keys_to_clear = [
            "ml_df_result", "ml_numerical_columns",
            "insights_done", "insights_data",
            "date_extracted", "date_extracted_cols",
            "revenue_created", "revenue_stats",
            "cat_handled", "cat_dropped", "cat_encoded",
            "df_after_cm",
            "fs_done", "fs_missing", "fs_preview",
            "selected_feature_cols", "selected_target_col", "df_for_ml",
            "pred_done", "pred_results", "pred_comparison",
            "pred_feature_cols", "pred_target_col",
            "pred_missing_before", "pred_train_size", "pred_test_size",
            "y_test", "best_model_name", "best_r2", "prediction_results",
            "feature_importance",
            "cluster_done", "cluster_cols", "cluster_inertia",
            "cluster_k_range", "cluster_suggested_k", "cluster_missing",
            "cluster_rows", "X_scaled_list", "X_cluster_dict",
            "cluster_profiles", "n_clusters", "silhouette",
            "cluster_cols_used", "df_clustered",
            "chosen_k_confirmed", "elbow_k_range",
            "llm_report",
            "industry", "goal", "audience",
            "outlier_done", "outlier_numerical_columns", "outlier_total_records",
            "outlier_zscore_count", "outlier_iqr_count", "outlier_iso_count",
            "outlier_zscore_flags", "outlier_iqr_flags", "outlier_iso_flags",
            "outlier_ml_df", "chosen_outlier_method", "chosen_outlier_count",
            "best_model_object", "scaler_object", "feature_means",
            "feature_mins", "feature_maxs", "best_model_type",
            "pred_interface_result", "pred_interface_low",
            "pred_interface_high", "pred_interface_inputs",
            "encoding_maps",
            "clustering_run", "cluster_cols_selected", "elbow_inertia",
            "chosen_k_confirmed",
            "df_cleaned",
            "needs_rerun",
            "missing_handled_done","domain_rules_done"
            

        ]

        for key in keys_to_clear:
            st.session_state.pop(key, None)

        st.session_state["last_uploaded_file"] = current_file

    # Dataset Loading
    df = pd.read_csv(uploaded_file)
    raw_df = df.copy()

    if "df_cleaned" in st.session_state:
        df = st.session_state["df_cleaned"]
    
        







    
    

    initial_rows = raw_df.shape[0]
    initial_missing = raw_df.isnull().sum().sum()

    st.success(" Dataset uploaded successfully!")

    st.header("Business Context")

    industry = st.selectbox("What industry is this data from?", 
    ["E-Commerce / Retail", "Grocery / FMCG", "Fashion", "Electronics", "B2B / SaaS"])

    goal = st.selectbox("What is your main business goal?",
    ["Increase revenue", "Reduce customer churn", "Detect fraud/anomalies", 
     "Understand customer segments", "Optimize inventory"])
    
    audience = st.selectbox("Who will read this report?",
    ["Business analyst", "CEO / Management", "Operations team", "Data team"])

    # Save to session state so LLM phase can access them
    st.session_state["industry"] = industry
    st.session_state["goal"] = goal
    st.session_state["audience"] = audience

    st.subheader("Baseline Dataset State (Before Cleaning)")
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", initial_rows)
    col2.metric("Total Missing Values", initial_missing)

    with st.expander("🔍 View Dataset Preview"):
        st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    # Data Profiling
    st.header("Data Profiling")

    st.subheader("Column Names")
    st.write(list(df.columns))

    st.subheader("Data Types")
    st.dataframe(df.dtypes.to_frame(name="Data Type"))

    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    st.subheader("Detected Attribute Types")
    st.write("🔢 Numerical Attributes:", numerical_columns)
    st.write("🔠 Categorical Attributes:", categorical_columns)

           

    # Missing Value Analysis
    st.subheader("Missing Value Analysis")

    st.info("""
This phase analyzes missing values in the dataset.

The system calculates:
            
• Total missing values per column  
• Percentage of missing data  

Columns with missing values above the defined threshold (30%) are flagged as high-risk,
as they may reduce data reliability and affect analysis results.
""")

    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100

    missing_summary = pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_Percentage": missing_percentage.round(2)
    })

    st.dataframe(missing_summary)

    high_missing_threshold = 30
    high_missing_columns = missing_summary[
        missing_summary["Missing_Percentage"] > high_missing_threshold
    ]

    if not high_missing_columns.empty:
        st.warning("⚠ Columns with High Missing Values (>30%)")
        st.dataframe(high_missing_columns)
    else:
        st.success("No columns exceed the high missingness threshold.")

    # Missing Value Heatmap
    st.subheader("Missing Value Heatmap")

    st.caption("""
    ℹ️ This heatmap shows where missing values are located across your dataset.
    
    • Dark/coloured cells → missing value at that position
               
    • White/light cells   → value exists
    
    Patterns in missing data are important:
               
    • Random missingness → safe to fill with median or drop
               
    • Clustered missingness → may indicate a data collection problem
    """)

    with st.expander("🔗 View Missing Value Heatmap"):
        missing_total = df.isnull().sum().sum()

        if missing_total == 0:
            st.success("✅ No missing values found in the dataset. Heatmap not needed.")
        else:
            # Limit to first 100 rows for performance
            sample_size = min(100, len(df))
            df_sample = df.iloc[:sample_size]

            # Build binary missing matrix (1 = missing, 0 = present)
            missing_matrix = df_sample.isnull().astype(int)

            fig_miss, ax_miss = plt.subplots(
                figsize=(max(8, len(df.columns) * 0.8), 5)
            )

            sns.heatmap(
                missing_matrix,
                cbar=False,
                cmap="Reds",
                ax=ax_miss,
                yticklabels=False,
                linewidths=0
            )

            ax_miss.set_title(
                f"Missing Value Map — First {sample_size} Rows",
                fontsize=12
            )
            ax_miss.set_xlabel("Columns")
            ax_miss.set_ylabel("Rows")
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_miss)
            plt.close()

            st.caption(f"ℹ️ Showing first {sample_size} rows for performance. Red cells indicate missing values.")

            # Missing value summary per column
            st.subheader("Missing Value Summary")

            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df) * 100).round(2)

            missing_summary = pd.DataFrame({
                "Column": missing_counts.index,
                "Missing Count": missing_counts.values,
                "Missing %": missing_pct.values,
                "Status": [
                    "🔴 High — above 30%" if p > 30
                    else "🟡 Moderate — 5% to 30%" if p > 5
                    else "🟢 Low — under 5%"
                    for p in missing_pct.values
                ]
            })

            missing_summary = missing_summary[missing_summary["Missing Count"] > 0]

            if len(missing_summary) > 0:
                st.dataframe(missing_summary, use_container_width=True)
                st.caption("""
                💡 Recommendation:
                🔴 High missingness (>30%) → consider dropping the column
                🟡 Moderate missingness (5–30%) → fill with median or mode
                🟢 Low missingness (<5%) → safe to fill with median or mode
                """)
            else:
                st.success("✅ No missing values detected in any column.")

        # Duplicate Detection
    st.subheader("Duplicate Record Analysis")

    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100

    col1, col2 = st.columns(2)
    col1.metric("Total Duplicate Rows", duplicate_count)
    col2.metric("Duplicate Percentage", f"{duplicate_percentage:.2f}%")

    if duplicate_count > 0:
        duplicate_rows = df[df.duplicated(keep=False)]
        st.dataframe(duplicate_rows.head())
    else:
        st.success("No duplicate records found.")

    # Statistical Profiling
    st.subheader("Basic Statistical Profiling")

    if numerical_columns:
        stats_df = df[numerical_columns].describe()
        st.dataframe(stats_df)

        min_max_df = df[numerical_columns].agg(["min", "max"])
        st.subheader("MinMax Values (Range Inspection)")
        st.dataframe(min_max_df)
    else:
        st.info("No numerical columns detected.")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    st.caption("""
    ℹ️ A correlation heatmap shows how strongly pairs of numeric columns 
    are related to each other.
    
    • Values close to  1.0 → strong positive relationship (both increase together)
               
    • Values close to -1.0 → strong negative relationship (one increases, other decreases)
               
    • Values close to  0.0 → no relationship
    
    This helps identify which columns may be redundant or strongly linked
    before building machine learning models.
    """)

    with st.expander("🔗 View Correlation Heatmap"):
        numerical_columns_heatmap = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if len(numerical_columns_heatmap) < 2:
            st.info("ℹ️ Not enough numeric columns to generate a correlation heatmap.")
        else:
            corr_matrix = df[numerical_columns_heatmap].corr()

            fig_corr, ax_corr = plt.subplots(
                figsize=(max(6, len(numerical_columns_heatmap)),
                         max(4, len(numerical_columns_heatmap) - 1))
            )

            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax_corr,
                annot_kws={"size": 9}
            )

            ax_corr.set_title("Correlation Matrix — Numeric Columns", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_corr)
            plt.close()

            # Highlight strong correlations
            st.subheader("Strongly Correlated Column Pairs")
            st.caption("ℹ️ Column pairs with correlation above 0.8 or below -0.8 are highlighted. High correlation between two feature columns can affect model performance.")

            strong_pairs = []
            cols = corr_matrix.columns.tolist()

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) >= 0.8:
                        strong_pairs.append({
                            "Column A": cols[i],
                            "Column B": cols[j],
                            "Correlation": round(val, 4),
                            "Relationship": "Strong Positive ↑↑" if val > 0 else "Strong Negative ↑↓"
                        })

            if strong_pairs:
                st.dataframe(pd.DataFrame(strong_pairs), use_container_width=True)
                st.caption("💡 Consider dropping one column from each strongly correlated pair when selecting features for prediction.")
            else:
                st.success("✅ No strongly correlated column pairs found (threshold: 0.8).")

    


        # Rule-Based Cleaning
    st.header("Rule-Based Data Cleaning")

    st.subheader("Duplicate Handling")

    rows_before = df.shape[0]
    duplicate_count = df.duplicated().sum()

    st.write(f"Total Rows Before Cleaning: {rows_before}")
    st.write(f"Duplicate Records Detected: {duplicate_count}")

    if duplicate_count > 0:
        if st.toggle("Remove duplicate records"):

            df = df.drop_duplicates()

            rows_after = df.shape[0]
            removed_duplicates = rows_before - rows_after

            st.success("Duplicate removal completed.")
            
            st.write(f"Total Rows After Cleaning: {rows_after}")
            st.write(f"Duplicate Records Removed: {removed_duplicates}")
    else:
        st.success("No duplicate records found.")
    
    st.session_state["df_cleaned"] = df

    st.subheader("Missing value Handling")

    st.info("""
This phase handles missing values using rule-based strategies.

The system will:
            
• Treat numerical and date fields as critical attributes  
• Remove rows with missing values in critical fields  
• Fill missing values in non-critical numerical fields using median  
• Fill missing values in categorical fields with 'Unknown'  

Users can review the strategy before applying changes.
""")

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    st.write("Detected Numerical Columns:", numerical_columns)
    st.write("Detected Categorical Columns:", categorical_columns)

    st.write("Missing Values Before Handling:")
    st.dataframe(df.isnull().sum())

    critical_fields = numerical_columns.copy()

    for col in df.columns:
        if col not in critical_fields:
            try:
                df[col] = pd.to_datetime(df[col])
                critical_fields.append(col)
            except:
                continue

    non_critical_fields = [col for col in df.columns if col not in critical_fields]

    st.write("Critical Fields:", critical_fields)
    st.write("Non-Critical Fields:", non_critical_fields)

    if st.toggle("Apply Missing Value Handling Rules"):
        if not st.session_state.get("missing_handled_done"):

            missing_before = df[critical_fields].isnull().sum().sum()
            rows_before = df.shape[0]

            df = df.dropna(subset=critical_fields)

            rows_after_drop = df.shape[0]
            removed_rows = rows_before - rows_after_drop

            for col in non_critical_fields:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna('Unknown')

            st.session_state["df_cleaned"] = df
            st.session_state["missing_handled_done"] = True

            missing_after = df.isnull().sum().sum()

            st.success("Missing value handling completed.")
            st.write("Rows removed due to critical missing values:", removed_rows)
            st.write("Total Missing Before:", missing_before)
            st.write("Total Missing After:", missing_after)

            st.subheader("Dataset Preview After Missing Handling")
            st.dataframe(df.head())



    # Domain Column Mapping
    st.header("Domain Column Mapping")

    st.write("Map dataset columns to domain fields")
    st.caption("Select dataset columns that represent domain fields. Leave as 'None' if the dataset does not contain the attribute.")

    column_options = ["None"] + list(df.columns)

    col1, col2 = st.columns(2)

    with col1:
        price_col = st.selectbox(
            "Select Price Column",
            column_options,
            index=0
        )

    with col2:
        qty_col = st.selectbox(
            "Select Quantity Column",
            column_options,
            index=0
        )

    st.header("Domain-Specific Validation Rules")

    st.info("""
This phase performs domain-specific validation checks on the dataset.

The system will:
            
• Detect negative price values  
• Detect zero or negative quantity values  
• Identify records that violate business rules  

Users can review the detected issues before applying cleaning.
""")

    neg_prices = 0
    invalid_qty = 0

    if price_col != "None":
        neg_prices = (df[price_col] < 0).sum()
        st.write(f"Negative prices detected: {neg_prices}")
    else:
        st.info("Price column not selected. Skipping price validation.")

    if qty_col != "None":
        invalid_qty = (df[qty_col] <= 0).sum()
        st.write(f"Zero or negative quantities detected: {invalid_qty}")
    else:
        st.info("Quantity column not selected. Skipping quantity validation.")

    apply_domain_rules = st.toggle("Apply Domain Validation Rules")

    if apply_domain_rules:
        if not st.session_state.get("domain_rules_done"):

            rows_before = df.shape[0]

            if price_col != "None":
                df = df[df[price_col] >= 0]

            if qty_col != "None":
                df = df[df[qty_col] > 0]

            rows_after = df.shape[0]
            removed_rows = rows_before - rows_after

            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            for col in categorical_columns:
                df[col] = df[col].str.strip().str.lower()

            st.session_state["df_cleaned"] = df
            st.session_state["domain_rules_done"] = True

            st.success("Domain-specific cleaning applied successfully!")
            st.write(f"Rows removed due to domain violations: {removed_rows}")

            st.subheader("Dataset Preview After Domain Cleaning")
            st.dataframe(df.head())

    # User Validation Summary
    st.header("Cleaning Summary of the Dataset")

    final_rows = df.shape[0]
    final_missing = df.isnull().sum().sum()

    rows_removed = initial_rows - final_rows
    missing_handled = initial_missing - final_missing

    

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Rows", initial_rows)
    col2.metric("Final Rows", final_rows)
    col3.metric("Rows Removed", rows_removed)

    st.write("")
    col4, col5, col6 = st.columns(3)
    col4.metric("Initial Missing", initial_missing)
    col5.metric("Final Missing", final_missing)
    col6.metric("Missing Handled", missing_handled)

    if "ml_df_result" in st.session_state:
        df = st.session_state["ml_df_result"]
    
    # Download Cleaned Dataset
    st.header("Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode('utf-8')
    
    cleaned_filename = f"cleaned_{current_file}"
    
    
    st.download_button(
        label="Download Cleaned Dataset as CSV",
        data=csv,
        file_name=cleaned_filename,
        mime="text/csv"
    )

   
    # OUTLIER DETECTION & ANALYSIS
    
    st.header("Outlier Detection & Analysis")

    st.info("""
    This phase detects unusual values in your dataset using three different methods.
    
    • Outliers are flagged — not automatically removed.
            
    • You can compare all three methods and choose which one to trust.
            
    • Removal is optional and always your decision.
    
    📌 In e-commerce data, outliers are not always errors.
    A very large order could be a legitimate bulk purchase from a VIP customer.
    Always review before removing.
    """)

    st.subheader("Method Explanations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Z-Score**")
        st.caption("""
        Measures how far each value is from 
        the mean in standard deviations.
        Values beyond ±3 are flagged.
        
        ✅ Good for: normally distributed data
                         
        ⚠️ Weakness: fails on skewed data
        """)

    with col2:
        st.markdown("**IQR (Interquartile Range)**")
        st.caption("""
        Flags values that fall below 
        Q1 − 1.5×IQR or above Q3 + 1.5×IQR.
        Based on the middle 50% of data.
        
        ✅ Good for: skewed data like Revenue
                   
        ⚠️ Weakness: may flag valid extremes
        """)

    with col3:
        st.markdown("**Isolation Forest**")
        st.caption("""
        ML-based method that isolates 
        anomalies using random decision trees.
        Looks at all columns together.
        
        ✅ Good for: complex multi-column patterns
                   
        ⚠️ Weakness: harder to interpret why
        """)

    if st.button("🚀 Run Outlier Detection"):

        from sklearn.ensemble import IsolationForest
        from scipy import stats
        import numpy as np

        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numerical_columns) == 0:
            st.error("❌ No numerical columns available for outlier detection.")
        else:
            ml_df = df[numerical_columns].copy()
            ml_df = ml_df.fillna(ml_df.median())

            # METHOD 1: Z-Score
            z_scores = np.abs(stats.zscore(ml_df))
            zscore_outlier_mask = (z_scores > 3).any(axis=1)
            zscore_flags = pd.Series((~zscore_outlier_mask).astype(int))
            zscore_flags = zscore_flags.map({1: 1, 0: -1})
            zscore_count = int(zscore_outlier_mask.sum())

            #  METHOD 2: IQR 
            Q1 = ml_df.quantile(0.25)
            Q3 = ml_df.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outlier_mask = ((ml_df < (Q1 - 1.5 * IQR)) | (ml_df > (Q3 + 1.5 * IQR))).any(axis=1)
            iqr_flags = pd.Series((~iqr_outlier_mask).astype(int))
            iqr_flags = iqr_flags.map({1: 1, 0: -1})
            iqr_count = int(iqr_outlier_mask.sum())

            # METHOD 3: Isolation Forest 
            iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.01,
                random_state=42
            )
            iso_flags = iso_forest.fit_predict(ml_df)
            iso_count = int((iso_flags == -1).sum())

            total_records = len(df)

            # Save to session state 
            st.session_state["outlier_done"] = True
            st.session_state["outlier_numerical_columns"] = numerical_columns
            st.session_state["outlier_total_records"] = total_records
            st.session_state["outlier_zscore_count"] = zscore_count
            st.session_state["outlier_iqr_count"] = iqr_count
            st.session_state["outlier_iso_count"] = iso_count
            st.session_state["outlier_zscore_flags"] = zscore_flags.tolist()
            st.session_state["outlier_iqr_flags"] = iqr_flags.tolist()
            st.session_state["outlier_iso_flags"] = iso_flags.tolist()
            st.session_state["outlier_ml_df"] = ml_df.to_dict()

            # Set default Outlier_Flag to Isolation Forest to keep backward compatibility
            df['Outlier_Flag'] = iso_flags
            st.session_state["ml_df_result"] = df
            st.session_state["ml_numerical_columns"] = numerical_columns

    #  Always display if outlier detection was run
    if st.session_state.get("outlier_done"):

        import numpy as np

        numerical_columns = st.session_state["outlier_numerical_columns"]
        total_records = st.session_state["outlier_total_records"]
        zscore_count = st.session_state["outlier_zscore_count"]
        iqr_count = st.session_state["outlier_iqr_count"]
        iso_count = st.session_state["outlier_iso_count"]

        #  Comparison Table 
        st.subheader("Method Comparison")

        comparison_data = {
            "Method": ["Z-Score", "IQR", "Isolation Forest"],
            "Outliers Detected": [zscore_count, iqr_count, iso_count],
            "Percentage": [
                f"{zscore_count/total_records*100:.2f}%",
                f"{iqr_count/total_records*100:.2f}%",
                f"{iso_count/total_records*100:.2f}%"
            ],
            "Type": ["Statistical", "Statistical", "Machine Learning"]
        }

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        st.write(f"**Total records analysed:** {total_records}")

        # Interpretation
        st.subheader("Interpretation")

        counts = {"Z-Score": zscore_count, "IQR": iqr_count, "Isolation Forest": iso_count}
        most_method = max(counts, key=counts.get)
        least_method = min(counts, key=counts.get)

        st.caption(f"""
        ℹ️ **{most_method}** detected the most outliers ({counts[most_method]} rows).
        **{least_method}** was the most conservative ({counts[least_method]} rows).
        
        • IQR is generally more reliable for e-commerce data because Revenue and Quantity 
        are typically right-skewed — IQR handles skewed distributions better than Z-Score.
        
        • Isolation Forest looks at all columns together and finds rows that are unusual 
        across multiple dimensions — not just single columns.
        
        • If IQR and Z-Score agree on a row being an outlier, it is more likely a genuine anomaly.
        """)

        #  Box Plot Visualization
        st.subheader("Box Plot Visualisation")

        st.caption("""
        ℹ️ Box plots show the distribution of each numeric column.
        Dots beyond the whiskers are potential outliers.
        The box covers the middle 50% of values (IQR).
        The line inside the box is the median.
        """)

        ml_df = pd.DataFrame(st.session_state["outlier_ml_df"])

        cols_to_plot = pd.DataFrame(st.session_state["outlier_ml_df"]).columns.tolist()[:6]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, col in enumerate(cols_to_plot):
            axes[i].boxplot(ml_df[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="steelblue", alpha=0.6))
            axes[i].set_title(col, fontsize=11)
            axes[i].set_ylabel("Value")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        #  User picks method 
        st.subheader("Choose Outlier Flag Method")

        st.caption("""
        ℹ️ Select which method's result to use as the Outlier_Flag column going forward.
        This flag is used in Insight Generation and downstream analysis.
        """)

        chosen_method = st.radio(
            "Which method do you want to use for outlier flagging?",
            ["Z-Score", "IQR", "Isolation Forest"],
            index=2,
            horizontal=True
        )

        if st.button("✅ Apply Chosen Method"):

            if "ml_df_result" in st.session_state:
                df_update = st.session_state["ml_df_result"].copy()
            else:
                df_update = df.copy()

            if chosen_method == "Z-Score":
                df_update['Outlier_Flag'] = st.session_state["outlier_zscore_flags"]
                chosen_count = zscore_count
            elif chosen_method == "IQR":
                df_update['Outlier_Flag'] = st.session_state["outlier_iqr_flags"]
                chosen_count = iqr_count
            else:
                df_update['Outlier_Flag'] = st.session_state["outlier_iso_flags"]
                chosen_count = iso_count

            st.session_state["ml_df_result"] = df_update
            st.session_state["chosen_outlier_method"] = chosen_method
            st.session_state["chosen_outlier_count"] = chosen_count

        if st.session_state.get("chosen_outlier_method"):
            chosen_method_display = st.session_state["chosen_outlier_method"]
            chosen_count_display = st.session_state["chosen_outlier_count"]
            st.success(f"✅ Outlier_Flag set using **{chosen_method_display}** — {chosen_count_display} outliers flagged.")

        # Sample outliers 
        st.subheader("Sample Flagged Outliers")
        st.caption("ℹ️ These are the first few rows flagged as outliers by your chosen method.")

        if "ml_df_result" in st.session_state:
            df_display = st.session_state["ml_df_result"]
            outlier_sample = df_display[df_display['Outlier_Flag'] == -1].head()
            if len(outlier_sample) > 0:
                st.dataframe(outlier_sample)
            else:
                st.info("No outliers found with current method.")

        # Optional Removal 
        st.subheader("Optional Outlier Removal")

        st.caption("""
        ℹ️ Removing outliers is optional and should be done carefully.
        Only remove if you believe the flagged rows are data errors , 
        not legitimate transactions. In e-commerce, large orders may be 
        valid bulk purchases from wholesale customers.
        """)

        remove_outliers = st.toggle(
            "Remove flagged outliers from dataset before analysis",
            value=False
        )

        if remove_outliers:
            if "ml_df_result" in st.session_state:
                df_before = st.session_state["ml_df_result"].copy()
                rows_before = df_before.shape[0]
                df_cleaned = df_before[df_before['Outlier_Flag'] == 1].copy()
                rows_after = df_cleaned.shape[0]
                rows_removed = rows_before - rows_after

                st.session_state["ml_df_result"] = df_cleaned
                st.warning(f"⚠️ {rows_removed} outlier rows removed. {rows_after} rows remaining.")
                st.caption("ℹ️ You can uncheck the box above to restore outliers if needed — but you will need to re-run the detection.")
        
        # Update main df from session state
        if "ml_df_result" in st.session_state:
            df = st.session_state["ml_df_result"]

    # Insight Generation
    st.header("Insight Generation")

    if "ml_df_result" in st.session_state and "df_after_cm" not in st.session_state:
        df = st.session_state.get("df_after_cm", df)

    st.subheader("Insight Configuration")

    column_options = ["None"] + list(df.columns)

    price_col = st.selectbox("Select Price Column (optional)", column_options, key="price_col")
    quantity_col = st.selectbox("Select Quantity Column (optional)", column_options, key="quantity_col")
    product_col = st.selectbox("Select Product Column (optional)", column_options, key="product_col")
    country_col = st.selectbox("Select Country Column (optional)", column_options, key="country_col")
    date_col = st.selectbox("Select Date Column (optional)", column_options, key="date_col")

    generate_insights = st.button("Generate Insights")

    if generate_insights:

        #  Calculate all values 
        final_rows = df.shape[0]
        final_missing = df.isnull().sum().sum()
        rows_removed = initial_rows - final_rows
        missing_handled = initial_missing - final_missing

        missing_before = raw_df.isnull().sum()
        missing_after = df.isnull().sum()
        missing_comparison = pd.DataFrame({
            "Missing Before Cleaning": missing_before,
            "Missing After Cleaning": missing_after
        })

        num_outliers = int((df['Outlier_Flag'] == -1).sum()) if 'Outlier_Flag' in df.columns else None

        # Price insights
        price_stats_with = df[price_col].describe().to_dict() if price_col != "None" else None
        price_stats_without = None
        if price_col != "None" and 'Outlier_Flag' in df.columns:
            df_no_outliers = df[df['Outlier_Flag'] == 1]
            price_stats_without = df_no_outliers[price_col].describe().to_dict()

        # Price histogram data
        price_hist_data = df[price_col].dropna().tolist() if price_col != "None" else None

        # Top products
        top_products = None
        if product_col != "None" and quantity_col != "None":
            top_products = (
                df.groupby(product_col)[quantity_col]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .to_dict()
            )

        # Top countries
        top_countries = df[country_col].value_counts().head(10).to_dict() if country_col != "None" else None

        # Avg quantity
        avg_quantity = None
        if quantity_col != "None" and 'Outlier_Flag' in df.columns:
            avg_quantity = float(df[df['Outlier_Flag'] == 1][quantity_col].mean())

        # Monthly sales
        monthly_sales_dict = None
        if date_col != "None":
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                monthly_sales = (
                    df[date_col]
                    .dt.to_period('M')
                    .value_counts()
                    .sort_index()
                )
                monthly_sales.index = monthly_sales.index.astype(str)
                monthly_sales_dict = monthly_sales.to_dict()
            except:
                monthly_sales_dict = None

        # Save everything to session state 
        st.session_state["insights_done"] = True
        st.session_state["insights_data"] = {
            "final_rows": final_rows,
            "final_missing": final_missing,
            "rows_removed": rows_removed,
            "missing_handled": missing_handled,
            "missing_comparison": missing_comparison.to_dict(),
            "num_outliers": num_outliers,
            "price_col": price_col,
            "price_stats_with": price_stats_with,
            "price_stats_without": price_stats_without,
            "price_hist_data": price_hist_data,
            "product_col": product_col,
            "quantity_col": quantity_col,
            "top_products": top_products,
            "country_col": country_col,
            "top_countries": top_countries,
            "avg_quantity": avg_quantity,
            "date_col": date_col,
            "monthly_sales": monthly_sales_dict,
        }

    #  Always display if insights were generated 
    if st.session_state.get("insights_done"):

        d = st.session_state["insights_data"]

        st.subheader("Data Quality Insights")
        st.write(f"Initial dataset rows: {initial_rows}")
        st.write(f"Final dataset rows: {d['final_rows']}")
        st.write(f"Rows removed during cleaning: {d['rows_removed']}")
        st.write(f"Initial missing values: {initial_missing}")
        st.write(f"Final missing values: {d['final_missing']}")
        st.write(f"Missing values handled: {d['missing_handled']}")

        if d["num_outliers"] is not None:
            st.write(f"Detected outliers: {d['num_outliers']}")

        st.subheader("Missing Value Comparison")
        st.dataframe(pd.DataFrame(d["missing_comparison"]))

        if d["price_stats_with"]:
            st.subheader("Price Distribution Insights")
            st.write("Statistics WITH outliers")
            st.dataframe(pd.DataFrame(d["price_stats_with"], index=["value"]).T)

            if d["price_stats_without"]:
                st.write("Statistics WITHOUT outliers")
                st.dataframe(pd.DataFrame(d["price_stats_without"], index=["value"]).T)

            if d["price_hist_data"]:
                fig, ax = plt.subplots()
                sns.histplot(d["price_hist_data"], bins=30, ax=ax)
                ax.set_title("Price Distribution")
                st.pyplot(fig)
                plt.close()
        else:
            st.info("Price column not selected. Skipping price insights.")

        if d["top_products"]:
            st.subheader("Top Products by Sales Volume")
            st.bar_chart(pd.Series(d["top_products"]))
        else:
            st.info("Product or Quantity column not selected. Skipping product insights.")

        if d["top_countries"]:
            st.subheader("Top Countries by Transaction Count")
            st.bar_chart(pd.Series(d["top_countries"]))
        else:
            st.info("Country column not selected. Skipping country insights.")

        if d["avg_quantity"] is not None:
            st.subheader("Average Order Quantity (Excluding Outliers)")
            st.write(f"Average order quantity: {d['avg_quantity']:.2f}")
        else:
            st.info("Quantity column not selected. Skipping quantity insights.")

        if d["monthly_sales"]:
            st.subheader("Transaction Trends Over Time")
            st.line_chart(pd.Series(d["monthly_sales"]))
        elif d["date_col"] != "None":
            st.warning("Date column could not be processed.")
        else:
            st.info("Date column not selected. Skipping time-based insights.")

        
    
    # COLUMN MANAGEMENT PHASE
    
    st.header("Column Management & Feature Engineering")
    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)

   

   

    st.info("""
    This phase prepares your dataset for machine learning.

    • Extract useful features from date columns.
            
    • Create a Revenue column from Quantity × Price
            
    • Encode or drop categorical columns
    """)

    # 1. DATE FEATURE EXTRACTION 
    st.subheader("Date Feature Extraction")

    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                date_columns.append(col)
            except:
                pass
        elif 'datetime' in str(df[col].dtype):
            date_columns.append(col)

    if date_columns:
        selected_date_col = st.selectbox(
            "Select date column to extract features from",
            ["None"] + date_columns
        )

        if selected_date_col != "None":
            extract_options = st.multiselect(
                "Select features to extract",
                ["Month", "Day", "DayOfWeek", "Hour", "IsWeekend"],
                default=["Month", "DayOfWeek"]
            )

            if st.button("Extract Date Features"):
                df[selected_date_col] = pd.to_datetime(df[selected_date_col])

                if "Month" in extract_options:
                    df["Month"] = df[selected_date_col].dt.month.astype("int64")
                if "Day" in extract_options:
                    df["Day"] = df[selected_date_col].dt.day.astype("int64")
                if "DayOfWeek" in extract_options:
                    df["DayOfWeek"] = df[selected_date_col].dt.dayofweek.astype("int64")
                if "Hour" in extract_options:
                    df["Hour"] = df[selected_date_col].dt.hour.astype("int64")
                if "IsWeekend" in extract_options:
                    df["IsWeekend"] = df[selected_date_col].dt.dayofweek.isin([5, 6]).astype("int64")

                st.session_state.get("df_after_cm", df)
                st.session_state["date_extracted"] = True
                st.session_state["date_extracted_cols"] = extract_options
                st.rerun()

    else:
        st.info("No date columns detected in your dataset.")

    # Always show date extraction result
    if st.session_state.get("date_extracted"):
        st.success(f"✅ Date features extracted: {st.session_state['date_extracted_cols']}")

    # 2. REVENUE COLUMN CREATION 
    st.subheader("Revenue Feature Creation")

    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    col_options = ["None"] + numeric_cols

    default_qty = qty_col if qty_col != "None" and qty_col in numeric_cols else "None"
    default_price = price_col if price_col != "None" and price_col in numeric_cols else "None"

    col1, col2 = st.columns(2)
    with col1:
        qty_col_fe = st.selectbox(
            "Select Quantity column",
            col_options,
            index=col_options.index(default_qty)
        )
    with col2:
        price_col_fe = st.selectbox(
            "Select Price column",
            col_options,
            index=col_options.index(default_price)
        )

    if qty_col_fe != "None" and price_col_fe != "None":
        st.write(f"📌 Revenue = `{qty_col_fe}` × `{price_col_fe}`")
        st.caption("ℹ️ Revenue is useful for business insights and clustering. For prediction, consider using Quantity as your target , it avoids data leakage and represents a real forecasting problem.")

        if st.button("Create Revenue Column"):
            if "df_after_cm" in st.session_state:
                df = st.session_state.get("df_after_cm", df)
            df["Revenue"] = df[qty_col_fe] * df[price_col_fe]
            st.session_state["df_after_cm"] = df
            st.session_state["revenue_created"] = True
            st.session_state["revenue_stats"] = df["Revenue"].describe().to_dict()
            st.rerun()
    else:
        st.info("Select both Quantity and Price columns to create Revenue.")

    # Always show revenue result
    if st.session_state.get("revenue_created"):
        st.success("✅ Revenue column created successfully!")
        st.dataframe(pd.DataFrame(st.session_state["revenue_stats"], index=["value"]).T)
    
    # 3. CATEGORICAL COLUMN HANDLING 
    st.subheader("Categorical Column Handling")

    st.write("For each categorical column, choose to **Encode** it or **Drop** it.")

    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols:
        col_decisions = {}

        for col in categorical_cols:
            unique_count = df[col].nunique()

            if unique_count > 50:
                recommendation = "⚠️ High cardinality — recommended: Drop"
                default_choice = "Drop"
            else:
                recommendation = "✅ Low cardinality — recommended: Encode"
                default_choice = "Encode (Label)"

            st.write(f"**{col}** — {unique_count} unique values — {recommendation}")
            decision = st.radio(
                f"Action for `{col}`",
                ["Keep as-is", "Encode (Label)", "Drop"],
                index=["Keep as-is", "Encode (Label)", "Drop"].index(default_choice),
                horizontal=True,
                key=f"cat_{col}"
            )
            col_decisions[col] = decision

        if st.button("Apply Column Decisions"):
            from sklearn.preprocessing import LabelEncoder

            if "df_after_cm" in st.session_state:
                df = st.session_state.get("df_after_cm", df)

            dropped = []
            encoded = []
            encoding_maps = {}

            for col, decision in col_decisions.items():
                if decision == "Drop":
                    df = df.drop(columns=[col])
                    dropped.append(col)
                elif decision == "Encode (Label)":
                    le = LabelEncoder()
                    le.fit(df[col].astype(str))

                    encoding_maps[col] = dict(
                        zip(le.classes_, le.transform(le.classes_).tolist())
                    )

                    df[col] = le.transform(df[col].astype(str))
                    encoded.append(col)

            st.session_state["df_after_cm"] = df = df
            st.session_state["cat_handled"] = True
            st.session_state["cat_dropped"] = dropped
            st.session_state["cat_encoded"] = encoded
            st.session_state["encoding_maps"] = encoding_maps
            st.rerun()

    else:
        st.info("No categorical columns remaining in the dataset.")

    # Always show categorical handling result
    if st.session_state.get("cat_handled"):
        st.success("✅ Column decisions applied!")

        if st.session_state.get("cat_dropped"):
            st.write(f"Dropped columns: {st.session_state['cat_dropped']}")

        if st.session_state.get("cat_encoded"):
            st.write(f"Encoded columns: {st.session_state['cat_encoded']}")

            if "encoding_maps" in st.session_state and st.session_state["encoding_maps"]:
                st.subheader("Encoding Reference Table")
                st.caption("""
                ℹ️ This table shows how each category was converted to a number.
                Keep this in mind when using the Prediction Interface ,
                the system will show you a dropdown with original labels automatically.
                """)
                for col, mapping in st.session_state["encoding_maps"].items():
                    st.write(f"**{col}:**")
                    mapping_df = pd.DataFrame(
                        list(mapping.items()),
                        columns=["Original Label", "Encoded Number"]
                    )
                    st.dataframe(mapping_df, use_container_width=True)

    # Always update df from session state at the end
    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)



    
    
    # FEATURE SELECTION PHASE
  
    st.header("Feature Selection")

    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)

   


    st.info("""
    Select the columns your model will learn from (X features),
    and the column you want to predict (Y target).
    
    • Only numeric columns are shown ... make sure you completed Column Management first.
            
    • Your target column should be what you want to predict (e.g. Quantity).
    """)

    # Always use latest df from Column Management if available
    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)

    # Get numeric columns only
    available_numeric_cols = df.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()

    if len(available_numeric_cols) < 2:
        st.warning("⚠️ Not enough numeric columns for feature selection. Please complete the Column Management phase first.")
    else:

        # Target column selection
        st.subheader("Select Target Column (Y)")

        st.caption("""
        ℹ️ Prediction tip:
        Choose a target column that is NOT directly calculated from your other feature columns.
        Good targets for e-commerce: Quantity
        Avoid: Revenue if Quantity and UnitPrice are both features , this causes data leakage.
        """)

        target_col = st.selectbox(
            "What do you want to predict?",
            available_numeric_cols
        )

        # Feature columns selection
        st.subheader("Select Feature Columns (X)")
        remaining_cols = [col for col in available_numeric_cols if col != target_col]

        feature_cols = st.multiselect(
            "Select columns the model will learn from",
            remaining_cols,
            default=remaining_cols
        )

        if len(feature_cols) == 0:
            st.warning("⚠️ Please select at least one feature column.")
        else:

            if st.button("✅ Confirm Feature Selection"):

                preview_cols = feature_cols + [target_col]
                missing_in_selection = df[preview_cols].isnull().sum().sum()

                # Save everything to session state
                st.session_state["selected_feature_cols"] = feature_cols
                st.session_state["selected_target_col"] = target_col
                st.session_state["df_for_ml"] = df[preview_cols].copy()
                st.session_state["fs_done"] = True
                st.session_state["fs_missing"] = int(missing_in_selection)
                st.session_state["fs_preview"] = df[preview_cols].head().to_dict()

    # Always display if feature selection was confirmed
    if st.session_state.get("fs_done"):

        st.write(f"**Target (Y):** `{st.session_state['selected_target_col']}`")
        st.write(f"**Features (X):** {st.session_state['selected_feature_cols']}")

        st.subheader("Dataset Preview — Selected Features")
        st.dataframe(pd.DataFrame(st.session_state["fs_preview"]))

        if st.session_state["fs_missing"] > 0:
            st.warning(f"⚠️ {st.session_state['fs_missing']} missing values found. They will be filled with median automatically before training.")
        else:
            st.success("✅ No missing values in selected columns. Ready for training.")

        st.success("✅ Feature selection confirmed. Scroll down to the Prediction phase.")

    
    # PREDICTION PHASE (AutoML)
   
    st.header("AutoML — Prediction Phase")

    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)

    

    st.info("""
    This phase automatically trains 3 machine learning models on your prepared dataset,
    compares their performance, and identifies the best model for your prediction task.
    
    Make sure you have completed Feature Selection before running this phase.
    """)

    if "df_for_ml" not in st.session_state or "selected_feature_cols" not in st.session_state:
        st.warning("⚠️ Please complete the Feature Selection phase first.")
    else:

        if st.button("🚀 Run Prediction Models"):

            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from xgboost import XGBRegressor
            import numpy as np

            #  STEP 1: Load data 
            df_ml = st.session_state["df_for_ml"].copy()
            feature_cols = st.session_state["selected_feature_cols"]
            target_col = st.session_state["selected_target_col"]

            X = df_ml[feature_cols]
            y = df_ml[target_col]

            # STEP 2: Fill missing values
            missing_before = X.isnull().sum().sum()
            X = X.fillna(X.median())
            y = y.fillna(y.median())

            #  STEP 3: Train/test split 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # STEP 4: Scaling for Linear Regression 
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # STEP 5: Train all 3 models 
            results = {}

            with st.spinner("Training Linear Regression..."):
                lr = LinearRegression()
                lr.fit(X_train_scaled, y_train)
                y_pred_lr = lr.predict(X_test_scaled)
                results["Linear Regression"] = {
                    "r2": r2_score(y_test, y_pred_lr),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_lr))),
                    "mae": float(mean_absolute_error(y_test, y_pred_lr)),
                    "predictions": y_pred_lr.tolist(),
                    "coef": lr.coef_.tolist()
                }

            with st.spinner("Training Random Forest..."):
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                results["Random Forest"] = {
                    "r2": r2_score(y_test, y_pred_rf),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
                    "mae": float(mean_absolute_error(y_test, y_pred_rf)),
                    "predictions": y_pred_rf.tolist(),
                    "feature_importance": rf.feature_importances_.tolist()
                }

            with st.spinner("Training XGBoost..."):
                xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                xgb.fit(X_train, y_train)
                y_pred_xgb = xgb.predict(X_test)
                results["XGBoost"] = {
                    "r2": r2_score(y_test, y_pred_xgb),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
                    "mae": float(mean_absolute_error(y_test, y_pred_xgb)),
                    "predictions": y_pred_xgb.tolist(),
                    "feature_importance": xgb.feature_importances_.tolist()
                }

            # Find best model
            best_model_name = max(results, key=lambda x: results[x]["r2"])

            #  Save everything to session state 
            st.session_state["pred_done"] = True
            st.session_state["pred_results"] = results
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_r2"] = results[best_model_name]["r2"]
            st.session_state["pred_feature_cols"] = feature_cols
            st.session_state["pred_target_col"] = target_col
            st.session_state["pred_missing_before"] = int(missing_before)
            st.session_state["pred_train_size"] = X_train.shape[0]
            st.session_state["pred_test_size"] = X_test.shape[0]
            st.session_state["y_test"] = y_test.tolist()

            # Save comparison table
            comparison_data = {
                "Model": list(results.keys()),
                "R² Score": [round(results[m]["r2"], 4) for m in results],
                "RMSE": [round(results[m]["rmse"], 4) for m in results],
                "MAE": [round(results[m]["mae"], 4) for m in results]
            }
            st.session_state["pred_comparison"] = comparison_data
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_r2"] = results[best_model_name]["r2"]
            st.session_state["prediction_results"] = comparison_data

            # Save model objects for prediction interface 
            if best_model_name == "Random Forest":
                st.session_state["best_model_object"] = rf
            elif best_model_name == "XGBoost":
                st.session_state["best_model_object"] = xgb
            else:
                st.session_state["best_model_object"] = lr

            st.session_state["scaler_object"] = scaler
            st.session_state["feature_means"] = X_train.mean().to_dict()
            st.session_state["feature_mins"] = X_train.min().to_dict()
            st.session_state["feature_maxs"] = X_train.max().to_dict()
            st.session_state["best_model_type"] = best_model_name

    #  Always display if prediction was run 
    if st.session_state.get("pred_done"):

        results = st.session_state["pred_results"]
        best_model_name = st.session_state["best_model_name"]
        feature_cols = st.session_state["pred_feature_cols"]
        target_col = st.session_state["pred_target_col"]
        y_test = st.session_state["y_test"]

        # Step 1 — Data overview
        st.subheader("Step 1 — Data Overview")
        st.write(f"**Features (X):** {len(feature_cols)} columns")
        st.write(f"**Target (Y):** `{target_col}`")

        if st.session_state["pred_missing_before"] > 0:
            st.write(f"ℹ️ {st.session_state['pred_missing_before']} missing values were filled using median values.")

        # Step 2 — Train/test split
        st.subheader("Step 2 — Train / Test Split")
        st.write(f"**Training set:** {st.session_state['pred_train_size']} rows (80%) — the model learns from this")
        st.write(f"**Testing set:** {st.session_state['pred_test_size']} rows (20%) — the model is evaluated on this")
        st.caption("ℹ️ We keep the test set completely separate during training so the evaluation is honest..the model never sees these rows until we measure its performance.")

        # Step 3 — Scaling note
        st.subheader("Step 3 — Feature Scaling")
        st.caption("ℹ️ Linear Regression is sensitive to column scales so scaling is applied for it. Random Forest and XGBoost handle different scales naturally so they use original data.")

        # Step 4 — Model comparison
        st.subheader("Step 4 — Model Comparison")
        comparison_df = pd.DataFrame(st.session_state["pred_comparison"])
        st.dataframe(comparison_df, use_container_width=True)
        st.caption("""
        ℹ️ How to read these metrics:
        - **R² Score** — how much of the pattern the model explains. Closer to 1.0 is better.
        - **RMSE** — average prediction error in the same unit as your target. Lower is better.
        - **MAE** — similar to RMSE but less sensitive to large errors. Lower is better.
        """)

        # Step 5 — R² interpretation
        st.subheader("R² Score Interpretation")
        for model_name, metrics in results.items():
            r2 = metrics["r2"]
            if r2 >= 0.8:
                st.write(f"**{model_name}:** R² = {r2:.4f} — ✅ Strong model")
            elif r2 >= 0.5:
                st.write(f"**{model_name}:** R² = {r2:.4f} — ⚠️ Moderate model")
            else:
                st.write(f"**{model_name}:** R² = {r2:.4f} — ❌ Weak model — consider selecting different features")

        # Step 6 — Best model
        st.subheader("Best Model")
        best_r2 = results[best_model_name]["r2"]
        st.success(f"🏆 Best Model: **{best_model_name}** — R² = {best_r2:.4f}")
        st.write(f"This model explains **{best_r2*100:.1f}%** of the variation in `{target_col}`.")

        # Step 7 — Feature importance
        st.subheader("Feature Importance")
        st.caption("ℹ️ Feature importance shows which columns influenced predictions the most. A higher bar means that column had more impact on the model's decisions.")

        if best_model_name in ["Random Forest", "XGBoost"]:
            importance_values = results[best_model_name]["feature_importance"]
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importance_values
            }).sort_values("Importance", ascending=False)

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
            ax1.set_xlabel("Importance Score")
            ax1.set_title(f"Feature Importance — {best_model_name}")
            ax1.invert_yaxis()
            st.pyplot(fig1)
            plt.close()

        else:
            coef_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient": results["Linear Regression"]["coef"]
            }).sort_values("Coefficient", ascending=False)
            st.dataframe(coef_df, use_container_width=True)

        # Step 8 — Actual vs Predicted
        st.subheader("Actual vs Predicted")
        st.caption("ℹ️ Each dot represents one transaction from the test set. Points close to the diagonal line mean the prediction was accurate. Scattered points mean the model struggled with those values.")

        best_predictions = results[best_model_name]["predictions"]

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.scatter(y_test, best_predictions, alpha=0.4, color="steelblue", s=20)
        ax2.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            "r--", linewidth=1.5, label="Perfect prediction"
        )
        ax2.set_xlabel(f"Actual {target_col}")
        ax2.set_ylabel(f"Predicted {target_col}")
        ax2.set_title(f"Actual vs Predicted — {best_model_name}")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

        

        # Step 9 — Residuals Plot
        st.subheader("Residuals Plot")

        st.caption("""
        ℹ️ A residual is the difference between the actual value and the predicted value.
        
        Residual = Actual − Predicted
        
        • Points scattered randomly around the zero line → model is performing well
                   
        • Pattern or curve in residuals → model is missing something (non-linearity)
                   
        • Residuals getting larger as prediction increases → model struggles with high values
        
        A good model has residuals randomly scattered around zero with no clear pattern.
        """)

        import numpy as np

        best_predictions = results[best_model_name]["predictions"]
        residuals = [actual - predicted for actual, predicted in zip(y_test, best_predictions)]

        fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Residuals vs Predicted 
        axes[0].scatter(
            best_predictions,
            residuals,
            alpha=0.4,
            color="steelblue",
            s=20
        )
        axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Zero line")
        axes[0].set_xlabel(f"Predicted {target_col}")
        axes[0].set_ylabel("Residual (Actual − Predicted)")
        axes[0].set_title(f"Residuals vs Predicted — {best_model_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        #  Plot 2: Residuals Distribution 
        axes[1].hist(residuals, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
        axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero line")
        axes[1].set_xlabel("Residual Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residuals Distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Residuals interpretation 
        st.subheader("Residuals Interpretation")

        residuals_array = np.array(residuals)
        mean_residual = float(np.mean(residuals_array))
        std_residual = float(np.std(residuals_array))
        max_residual = float(np.max(np.abs(residuals_array)))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Residual", f"{mean_residual:.4f}")
            st.caption("Should be close to 0 for an unbiased model")

        with col2:
            st.metric("Std of Residuals", f"{std_residual:.4f}")
            st.caption("Lower means more consistent predictions")

        with col3:
            st.metric("Max Absolute Error", f"{max_residual:.4f}")
            st.caption("Largest single prediction error")

        if abs(mean_residual) < std_residual * 0.1:
            st.success("✅ Mean residual is close to zero — the model is unbiased.")
        else:
            st.warning("⚠️ Mean residual is not close to zero — the model may be systematically over or under predicting.")

        if len(set([1 if r > 0 else -1 for r in residuals])) == 2:
            pos = sum(1 for r in residuals if r > 0)
            neg = sum(1 for r in residuals if r < 0)
            balance = min(pos, neg) / max(pos, neg)
            if balance > 0.8:
                st.success(f"✅ Residuals are balanced — {pos} positive, {neg} negative. Model predicts evenly.")
            else:
                st.warning(f"⚠️ Residuals are unbalanced — {pos} positive, {neg} negative. Model tends to over or under predict.")

            st.success("✅ Prediction phase complete. Scroll down to the Clustering phase.")

        # PREDICTION INTERFACE
      
        st.subheader("Make a New Prediction")

        st.info(f"""
        The best model **{best_model_name}** is ready to make predictions.
        Enter values for each feature below and click Predict to get a result.
        
        📌 Target to predict: `{target_col}`
        """)

        if "best_model_object" not in st.session_state:
            st.warning("⚠️ Model object not found. Please re-run the prediction models above.")
        else:
            feature_means  = st.session_state["feature_means"]
            feature_mins   = st.session_state["feature_mins"]
            feature_maxs   = st.session_state["feature_maxs"]
            encoding_maps  = st.session_state.get("encoding_maps", {})

            st.caption("""
            ℹ️ Default values are set to the mean of each feature from your training data.
            Categorical columns show a dropdown with original labels.
            Adjust values to match the scenario you want to predict.
            """)

            # Dynamically generate input fields
            input_values = {}
            cols_per_row = 3
            feature_list = feature_cols

            for i in range(0, len(feature_list), cols_per_row):
                row_features = feature_list[i:i + cols_per_row]
                row_cols = st.columns(len(row_features))

                for j, feat in enumerate(row_features):

                    mean_val = float(feature_means.get(feat, 0))
                    min_val  = float(feature_mins.get(feat, 0))
                    max_val  = float(feature_maxs.get(feat, mean_val * 3 + 1))

                    # Encoded categorical — show dropdown
                    if feat in encoding_maps:
                        options = list(encoding_maps[feat].keys())
                        selected = row_cols[j].selectbox(
                            f"{feat}",
                            options=options,
                            key=f"pred_input_{feat}"
                        )
                        input_values[feat] = encoding_maps[feat][selected]

                    # Month — slider 1-12
                    elif "month" in feat.lower():
                        input_values[feat] = row_cols[j].slider(
                            f"{feat}",
                            min_value=1,
                            max_value=12,
                            value=max(1, min(12, int(round(mean_val)))),
                            key=f"pred_input_{feat}"
                        )

                    # DayOfWeek — slider 0-6
                    elif "dayofweek" in feat.lower() or "day_of_week" in feat.lower():
                        input_values[feat] = row_cols[j].slider(
                            f"{feat} (0=Mon, 6=Sun)",
                            min_value=0,
                            max_value=6,
                            value=max(0, min(6, int(round(mean_val)))),
                            key=f"pred_input_{feat}"
                        )

                    # IsWeekend — radio 0 or 1
                    elif "weekend" in feat.lower() or "isweekend" in feat.lower():
                        input_values[feat] = row_cols[j].radio(
                            f"{feat}",
                            options=[0, 1],
                            format_func=lambda x: "Weekday (0)" if x == 0 else "Weekend (1)",
                            horizontal=True,
                            key=f"pred_input_{feat}"
                        )

                    # Hour — slider 0-23
                    elif "hour" in feat.lower():
                        input_values[feat] = row_cols[j].slider(
                            f"{feat}",
                            min_value=0,
                            max_value=23,
                            value=max(0, min(23, int(round(mean_val)))),
                            key=f"pred_input_{feat}"
                        )

                    # Integer column — check dtype from dataframe
                    elif str(st.session_state["df_for_ml"][feat].dtype) in ["int64", "int32"]:
                        input_values[feat] = row_cols[j].number_input(
                            f"{feat}",
                            min_value=int(min_val),
                            max_value=int(max_val * 2),
                            value=int(round(mean_val)),
                            step=1,
                            key=f"pred_input_{feat}"
                        )

                    # Default — decimal number input
                    else:
                        step_val = float(max(0.01, round((max_val - min_val) / 100, 4)))
                        input_values[feat] = row_cols[j].number_input(
                            f"{feat}",
                            min_value=float(min_val),
                            max_value=float(max_val * 2),
                            value=float(round(mean_val, 2)),
                            step=step_val,
                            key=f"pred_input_{feat}"
                        )

            # Predict button
            if st.button("🔮 Predict"):

                import numpy as np

                model      = st.session_state["best_model_object"]
                scaler     = st.session_state["scaler_object"]
                model_type = st.session_state["best_model_type"]

                # Build input array in correct feature order
                input_array = np.array([[input_values[f] for f in feature_cols]])

                # Apply scaling only for Linear Regression
                if model_type == "Linear Regression":
                    input_array = scaler.transform(input_array)

                # Make prediction
                prediction = float(model.predict(input_array)[0])

                # Confidence range for Random Forest
                confidence_low  = None
                confidence_high = None

                if model_type == "Random Forest":
                    tree_predictions = np.array([
                        tree.predict(input_array)[0]
                        for tree in model.estimators_
                    ])
                    std_pred        = float(np.std(tree_predictions))
                    confidence_low  = round(prediction - 1.96 * std_pred, 4)
                    confidence_high = round(prediction + 1.96 * std_pred, 4)

                # Save to session state
                st.session_state["pred_interface_result"] = prediction
                st.session_state["pred_interface_low"]    = confidence_low
                st.session_state["pred_interface_high"]   = confidence_high
                st.session_state["pred_interface_inputs"] = input_values.copy()

            # Always show prediction result
            if st.session_state.get("pred_interface_result") is not None:

                prediction  = st.session_state["pred_interface_result"]
                conf_low    = st.session_state["pred_interface_low"]
                conf_high   = st.session_state["pred_interface_high"]
                last_inputs = st.session_state["pred_interface_inputs"]

                st.markdown("---")
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label=f"Predicted {target_col}",
                        value=f"{prediction:.4f}"
                    )

                with col2:
                    if conf_low is not None and conf_high is not None:
                        st.metric(
                            label="95% Confidence Range",
                            value=f"{conf_low:.2f} — {conf_high:.2f}"
                        )
                    else:
                        st.metric(
                            label="Model Used",
                            value=best_model_name
                        )

                if conf_low is not None:
                    st.caption(f"""
                    ℹ️ The model predicts **{prediction:.4f}** for `{target_col}`.
                    Based on variation across all 100 decision trees in Random Forest,
                    the true value is expected to fall between **{conf_low:.2f}** and
                    **{conf_high:.2f}** with 95% confidence.
                    """)
                else:
                    st.caption(f"""
                    ℹ️ The model predicts **{prediction:.4f}** for `{target_col}`.
                    Confidence intervals are available when Random Forest is the best model.
                    """)

                # Show what was entered
                st.subheader("Input Values Used")

                # Show original labels for encoded columns
                display_inputs = {}
                for feat, val in last_inputs.items():
                    if feat in encoding_maps:
                        reverse_map = {v: k for k, v in encoding_maps[feat].items()}
                        display_inputs[feat] = f"{reverse_map.get(val, val)} (encoded: {val})"
                    else:
                        display_inputs[feat] = val

                st.dataframe(
                    pd.DataFrame([display_inputs]),
                    use_container_width=True
                )

                st.caption("💡 Change the input values above and click Predict again to explore different scenarios.")

        
        

  
    # CLUSTERING PHASE
    
    st.header("Clustering Phase")

    # Always use latest df
    if "df_after_cm" in st.session_state:
        df_cluster = st.session_state.get("df_after_cm", df).copy()
    else:
        df_cluster = df.copy()

    st.info("""
    Clustering groups your data into natural segments without needing a target column.
    The algorithm finds hidden patterns by itself.
    
    📌 Example: Customers who buy in bulk, premium buyers, occasional buyers , 
    the system discovers these groups automatically from your data.
    
    💡 For meaningful business clusters, select columns that represent value or 
    behaviour , like Revenue, Quantity, UnitPrice. Avoid using only date parts 
    like Month or DayOfWeek as clustering columns.
    """)

    

    # Get numeric columns only
    numeric_cols_cluster = df_cluster.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols_cluster) < 2:
        st.warning("⚠️ Not enough numeric columns for clustering. Please complete the Column Management phase first.")
    else:

        # STEP 1: Column Selection 
        st.subheader("Step 1 — Select Columns for Clustering")
        st.caption("ℹ️ Select at least 2 numeric columns. The algorithm will find natural groups based on these columns only.")

        cluster_cols = st.multiselect(
            "Select columns to cluster on",
            numeric_cols_cluster,
            default=numeric_cols_cluster[:3] if len(numeric_cols_cluster) >= 3 else numeric_cols_cluster
        )

        if len(cluster_cols) < 2:
            st.warning("⚠️ Please select at least 2 columns for clustering.")
        else:

            if st.button("🚀 Run Clustering"):
                st.session_state["clustering_run"] = True
                st.session_state["cluster_cols_selected"] = cluster_cols
                st.session_state.pop("elbow_inertia", None)
                st.session_state.pop("chosen_k_confirmed", None)

            if st.session_state.get("clustering_run"):

                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import silhouette_score
                from sklearn.decomposition import PCA
                import numpy as np

                cluster_cols = st.session_state["cluster_cols_selected"]

                #  Safety check — columns must exist in current dataset 
                valid_cols = [col for col in cluster_cols if col in df_cluster.columns]

                if len(valid_cols) < 2:
                    st.info("ℹ️ Previous clustering used columns that don't exist in the new dataset. Please run clustering again.")
                    st.session_state.pop("clustering_run", None)
                    st.session_state.pop("cluster_cols_selected", None)
                    st.session_state.pop("elbow_inertia", None)
                    st.session_state.pop("chosen_k_confirmed", None)

                else:
                    cluster_cols = valid_cols

                    # STEP 2: Prepare data
                    st.subheader("Step 2 — Data Preparation")

                    X_cluster = df_cluster[cluster_cols].copy()

                    missing_count = X_cluster.isnull().sum().sum()
                    if missing_count > 0:
                        X_cluster = X_cluster.fillna(X_cluster.median())
                        st.write(f"ℹ️ {missing_count} missing values filled with median before clustering.")

                    st.write(f"**Rows:** {X_cluster.shape[0]} | **Columns used:** {cluster_cols}")

                    # STEP 3: Scaling 
                    st.subheader("Step 3 — Scaling")

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_cluster)

                    st.caption("""
                    ℹ️ Why scaling is mandatory for clustering:
                    If one column ranges from 0–50,000 and another from 1–12,
                    the large column completely dominates the grouping.
                    Scaling brings all columns to the same range so every 
                    column contributes equally to finding the groups.
                    """)

                    st.success("✅ All columns scaled to equal range.")

                    # STEP 4: Elbow Method 
                    st.subheader("Step 4 — Finding the Best Number of Clusters (Elbow Method)")

                    st.caption("""
                    ℹ️ KMeans needs you to specify how many groups (K) to create.
                    The elbow method runs KMeans for K=2 to 10 and measures how 
                    tight the groups are (inertia). Where the curve bends like an 
                    elbow — that is your best K. After that point, adding more 
                    groups gives very little improvement.
                    """)

                    if "elbow_inertia" not in st.session_state:
                        inertia_values = []
                        k_range = range(2, 11)
                        for k in k_range:
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)
                            km.fit(X_scaled)
                            inertia_values.append(km.inertia_)
                        st.session_state["elbow_inertia"] = inertia_values
                        st.session_state["elbow_k_range"] = list(k_range)
                    else:
                        inertia_values = st.session_state["elbow_inertia"]
                        k_range = range(2, 11)

                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(list(k_range), inertia_values, marker="o", color="steelblue", linewidth=2)
                    ax1.set_xlabel("Number of Clusters (K)")
                    ax1.set_ylabel("Inertia (tightness of clusters)")
                    ax1.set_title("Elbow Method — Finding Best K")
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)
                    plt.close()

                    deltas = [inertia_values[i] - inertia_values[i+1] for i in range(len(inertia_values)-1)]
                    suggested_k = k_range.start + deltas.index(max(deltas))
                    suggested_k = max(2, min(suggested_k, 6))

                    st.write(f"💡 Suggested K based on elbow: **{suggested_k}**")

                    # STEP 5: User picks K 
                    st.subheader("Step 5 — Choose Number of Clusters")

                    chosen_k = st.slider(
                        "Select number of clusters (K)",
                        min_value=2,
                        max_value=8,
                        value=suggested_k,
                        key="chosen_k_slider"
                    )

                    st.caption(f"ℹ️ You selected K={chosen_k}. The algorithm will divide your data into {chosen_k} groups.")

                    if st.button("▶️ Apply K and Run KMeans"):
                        st.session_state["chosen_k_confirmed"] = chosen_k

                    if "chosen_k_confirmed" not in st.session_state:
                        st.info("👆 Select your K value above and click Apply K and Run KMeans to see results.")

                    else:
                        chosen_k = st.session_state["chosen_k_confirmed"]

                        # STEP 6: Run KMeans 
                        st.subheader("Step 6 — Running KMeans")

                        kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        df_cluster["Cluster"] = cluster_labels

                        st.success(f"✅ KMeans completed. {chosen_k} clusters found.")

                        # STEP 7: Silhouette Score 
                        st.subheader("Step 7 — Cluster Quality Check")

                        sil_score = silhouette_score(X_scaled, cluster_labels)
                        st.write(f"**Silhouette Score: {sil_score:.4f}**")

                        if sil_score >= 0.5:
                            st.success("✅ Strong clusters — the groups are well separated.")
                        elif sil_score >= 0.2:
                            st.warning("⚠️ Reasonable clusters — some overlap exists between groups.")
                        else:
                            st.error("❌ Weak clusters — groups overlap heavily. Try selecting different columns or a different K.")

                        st.caption("""
                        ℹ️ Silhouette Score measures how well separated the clusters are.
                                   
                        Score above 0.5 → Strong and distinct groups ✅
                                   
                        Score 0.2–0.5  → Reasonable groups ⚠️
                                   
                        Score below 0.2 → Groups are too similar ❌
                        """)

                        #  STEP 8: Cluster Size Distribution 
                        st.subheader("Step 8 — Cluster Size Distribution")

                        cluster_counts = df_cluster["Cluster"].value_counts().sort_index()

                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        ax2.bar(
                            [f"Cluster {i}" for i in cluster_counts.index],
                            cluster_counts.values,
                            color="steelblue"
                        )
                        ax2.set_xlabel("Cluster")
                        ax2.set_ylabel("Number of rows")
                        ax2.set_title("Cluster Size Distribution")
                        st.pyplot(fig2)
                        plt.close()

                        st.caption("""
                        ℹ️ Ideally clusters should be reasonably balanced.
                        A cluster with very few rows may represent outliers or rare cases.
                        A cluster with almost all rows may mean the grouping is not meaningful.
                        """)

                        # STEP 9: Cluster Profiles 
                        st.subheader("Step 9 — Cluster Profiles")

                        st.caption("""
                        ℹ️ This table shows the average values for each cluster.
                        This is the most important output — it tells you what 
                        makes each group different from the others.
                        """)

                        cluster_profile = df_cluster.groupby("Cluster")[cluster_cols].mean().round(2)
                        st.dataframe(cluster_profile, use_container_width=True)

                        st.subheader("Cluster Labels")
                        st.caption("ℹ️ Based on the profile values, here is a simple interpretation of each cluster:")

                        overall_mean = df_cluster[cluster_cols[0]].mean()
                        for cluster_id in range(chosen_k):
                            cluster_mean = cluster_profile.loc[cluster_id, cluster_cols[0]]
                            if cluster_mean > overall_mean * 1.3:
                                label = "🔴 High Value Group"
                            elif cluster_mean < overall_mean * 0.7:
                                label = "🔵 Low Value Group"
                            else:
                                label = " 🔵 Medium Value Group"
                            st.write(f"**Cluster {cluster_id}:** {label} — avg `{cluster_cols[0]}` = {cluster_mean:.2f}")

                        # STEP 10: Scatter Plot
                        st.subheader("Step 10 — Cluster Visualisation")

                        st.caption("""
                        ℹ️ Each dot represents one row in your dataset.
                        Dots of the same colour belong to the same cluster.
                        Well separated colour groups mean the clustering worked well.
                        If using more than 2 columns, PCA reduces them to 2D for 
                        visualisation only — the actual clustering was done on all 
                        selected columns.
                        """)

                        colors = ["steelblue", "coral", "green", "purple", "orange", "red", "brown", "pink"]

                        if len(cluster_cols) == 2:
                            fig3, ax3 = plt.subplots(figsize=(7, 5))
                            for cluster_id in range(chosen_k):
                                mask = cluster_labels == cluster_id
                                ax3.scatter(
                                    X_cluster[cluster_cols[0]][mask],
                                    X_cluster[cluster_cols[1]][mask],
                                    label=f"Cluster {cluster_id}",
                                    alpha=0.5,
                                    s=20,
                                    color=colors[cluster_id % len(colors)]
                                )
                            ax3.set_xlabel(cluster_cols[0])
                            ax3.set_ylabel(cluster_cols[1])
                            ax3.set_title("Cluster Scatter Plot")
                            ax3.legend()
                            st.pyplot(fig3)
                            plt.close()

                        else:
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            explained = pca.explained_variance_ratio_.sum() * 100

                            fig3, ax3 = plt.subplots(figsize=(7, 5))
                            for cluster_id in range(chosen_k):
                                mask = cluster_labels == cluster_id
                                ax3.scatter(
                                    X_pca[mask, 0],
                                    X_pca[mask, 1],
                                    label=f"Cluster {cluster_id}",
                                    alpha=0.5,
                                    s=20,
                                    color=colors[cluster_id % len(colors)]
                                )
                            ax3.set_xlabel("PCA Component 1")
                            ax3.set_ylabel("PCA Component 2")
                            ax3.set_title(f"Cluster Visualisation (PCA — {explained:.1f}% variance explained)")
                            ax3.legend()
                            st.pyplot(fig3)
                            plt.close()

                            st.caption(f"ℹ️ PCA reduced your {len(cluster_cols)} columns to 2D for visualisation. {explained:.1f}% of the original data variation is preserved in this plot.")

                        # STEP 11: Save to session state
                        st.session_state["cluster_profiles"] = cluster_profile.to_dict()
                        st.session_state["n_clusters"] = chosen_k
                        st.session_state["silhouette"] = round(sil_score, 4)
                        st.session_state["cluster_cols_used"] = cluster_cols
                        st.session_state["df_clustered"] = df_cluster

                        st.success("✅ Clustering complete. Scroll down to the LLM Business Summary phase.")

  
    # LLM BUSINESS SUMMARY PHASE
    
    st.header("LLM Business Summary & Recommendations")

    if "df_after_cm" in st.session_state:
        df = st.session_state.get("df_after_cm", df)
    

    st.info("""
    This phase uses Google Gemini AI to analyse your prediction and clustering results
    and generate a plain-English business report with actionable recommendations.
    
    Make sure you have completed at least one of the Prediction or Clustering phases before running this.
    """)

    # Check if required results exist in session state
    prediction_done = "best_model_name" in st.session_state and "best_r2" in st.session_state
    clustering_done = "cluster_profiles" in st.session_state and "n_clusters" in st.session_state

    if not prediction_done and not clustering_done:
        st.warning("⚠️ Please complete at least the Prediction or Clustering phase first.")
    else:

        # Show what results are available
        st.subheader("Results Available for Analysis")

        col1, col2 = st.columns(2)

        with col1:
            if prediction_done:
                st.success(f"✅ Prediction — {st.session_state['best_model_name']} (R² = {st.session_state['best_r2']:.4f})")
            else:
                st.warning("⚠️ Prediction phase not completed — will be excluded from report")

        with col2:
            if clustering_done:
                st.success(f"✅ Clustering — {st.session_state['n_clusters']} clusters (Silhouette = {st.session_state.get('silhouette', 'N/A')})")
            else:
                st.warning("⚠️ Clustering phase not completed — will be excluded from report")

        if st.button("Generate Business Report"):

            from google import genai

            # API KEY
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

            try:
                # Read business context from session state
                industry = st.session_state.get("industry", "E-Commerce / Retail")
                goal     = st.session_state.get("goal", "Understand customer segments")
                audience = st.session_state.get("audience", "Business analyst")

                # BUILD PROMPT
                prompt_parts = []
                prompt_parts.append(f"You are a senior business analyst writing a report for a {audience} in the {industry} industry.")
                prompt_parts.append(f"The business main goal is to: {goal}.")
                prompt_parts.append("Based on the following machine learning results, write a clear business report.")
                prompt_parts.append(f"Write specifically for a {audience} — adjust your language, tone and recommendations to suit this audience.")
                prompt_parts.append("Use simple language. Avoid technical jargon unless the audience is a Data team.")
                prompt_parts.append("")

                # Add prediction results if available
                if prediction_done:
                    prompt_parts.append("=== PREDICTION RESULTS ===")
                    prompt_parts.append(f"Best Model: {st.session_state['best_model_name']}")
                    prompt_parts.append(f"R² Score: {st.session_state['best_r2']:.4f} ({st.session_state['best_r2']*100:.1f}% accuracy)")
                    prompt_parts.append(f"Target Predicted: {st.session_state.get('selected_target_col', 'Unknown')}")
                    prompt_parts.append(f"Features Used: {st.session_state.get('selected_feature_cols', [])}")

                    if "prediction_results" in st.session_state:
                        pred_results = st.session_state["prediction_results"]
                        prompt_parts.append("Model Comparison:")
                        for i in range(len(pred_results["Model"])):
                            prompt_parts.append(f"  - {pred_results['Model'][i]}: R²={pred_results['R² Score'][i]}, RMSE={pred_results['RMSE'][i]}")
                    prompt_parts.append("")

                # Add clustering results if available
                if clustering_done:
                    prompt_parts.append("=== CLUSTERING RESULTS ===")
                    prompt_parts.append(f"Number of Clusters: {st.session_state['n_clusters']}")
                    prompt_parts.append(f"Silhouette Score: {st.session_state.get('silhouette', 'N/A')} (higher is better, max 1.0)")
                    prompt_parts.append(f"Columns Clustered On: {st.session_state.get('cluster_cols_used', [])}")
                    prompt_parts.append("Cluster Profiles (average values per cluster):")

                    cluster_profiles = st.session_state["cluster_profiles"]
                    cols_used        = st.session_state.get("cluster_cols_used", [])
                    n_clusters       = st.session_state["n_clusters"]

                    for cluster_id in range(n_clusters):
                        profile_line = f"  Cluster {cluster_id}: "
                        values = []
                        for col in cols_used:
                            if col in cluster_profiles:
                                val = cluster_profiles[col].get(cluster_id, "N/A")
                                values.append(f"{col}={val}")
                        profile_line += ", ".join(values)
                        prompt_parts.append(profile_line)
                    prompt_parts.append("")

                # Dynamically build report structure based on what was completed
                prompt_parts.append("=== REPORT STRUCTURE ===")
                prompt_parts.append("Write the report with ONLY these sections based on the data provided above:")
                prompt_parts.append("")

                section_number = 1

                if prediction_done:
                    prompt_parts.append(f"{section_number}. PREDICTION INSIGHTS")
                    prompt_parts.append("   Explain what the model learned and what it means for the business.")
                    prompt_parts.append("   Mention which features matter most if available.")
                    prompt_parts.append("")
                    section_number += 1

                if clustering_done:
                    prompt_parts.append(f"{section_number}. CUSTOMER SEGMENT ANALYSIS")
                    prompt_parts.append("   Describe each cluster in plain business language.")
                    prompt_parts.append("   Give each cluster a meaningful business name (e.g. Premium Buyers, Bulk Buyers).")
                    prompt_parts.append("   Explain what makes each group different.")
                    prompt_parts.append("")
                    section_number += 1

                prompt_parts.append(f"{section_number}. TOP 3 BUSINESS RECOMMENDATIONS")
                prompt_parts.append("   Give 3 specific and actionable recommendations based on the results above.")
                prompt_parts.append("   Each recommendation must directly reference the actual data findings provided.")
                prompt_parts.append("")
                section_number += 1

                prompt_parts.append(f"{section_number}. DATA IMPROVEMENT SUGGESTION")
                prompt_parts.append("   Suggest one type of additional data that would improve future analysis.")
                prompt_parts.append("")

                prompt_parts.append("IMPORTANT: Only write sections for the data provided above.")
                prompt_parts.append("Do NOT invent or assume prediction results if none were provided.")
                prompt_parts.append("Do NOT invent or assume clustering results if none were provided.")
                prompt_parts.append("Keep the entire report under 400 words. Be specific and practical.")

                full_prompt = "\n".join(prompt_parts)

                # CALL GEMINI API
                with st.spinner("Gemini is analysing your results... This may take a few seconds."):

                    client   = genai.Client(api_key=GEMINI_API_KEY)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=full_prompt
                    )
                    report_text = response.text

                # DISPLAY REPORT
                st.subheader("Generated Business Report")
                st.markdown(report_text)

                # Save report to session state
                st.session_state["llm_report"] = report_text

                # Download button
                st.download_button(
                    label="Download Business Report",
                    data=report_text,
                    file_name="business_report.txt",
                    mime="text/plain"
                )

                # Disclaimer
                st.caption("""
                ⚠️ This report is AI-generated based on your data results.
                Always validate recommendations with domain expertise
                before making real business decisions.
                """)

            except Exception as e:
                st.error(f"❌ Error calling Gemini API: {str(e)}")
                st.caption("Please check your API key and make sure google-genai is installed correctly.")
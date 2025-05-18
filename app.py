
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template
import os
from io import BytesIO
import base64
import numpy as np
from scipy.stats import chi2_contingency
import gunicorn
import logging
import matplotlib.gridspec as gridspec

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set pandas display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

dataset_titles = {
    1: "League",
    2: "Abalone",
    3: "Diamond",
    4: "TELECOM",
    5: "STOCKS"
}

# Clean data
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
    return df

# Generate context from CSV data
def generate_csv_context(df):
    try:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        context = [f"Dataset contains {len(df)} rows and {len(df.columns)} columns."]
        if num_cols:
            context.append(f"Numeric columns: {', '.join(num_cols)}")
            for col in num_cols:
                context.append(f"- {col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, "
                              f"Min={df[col].min():.2f}, Max={df[col].max():.2f}")
        if cat_cols:
            context.append(f"Categorical columns: {', '.join(cat_cols)}")
            for col in cat_cols:
                top_cat = df[col].value_counts().idxmax()
                top_count = df[col].value_counts().max()
                context.append(f"- {col}: Most frequent value='{top_cat}' ({top_count} occurrences)")
        return '\n'.join(context)
    except Exception as e:
        logger.error(f"Error generating CSV context: {str(e)}")
        return f"Error generating context: {str(e)}"

# Load data for a dataset
def load_data(dataset_num):
    try:
        base_path = os.getenv('DATASET_PATH', '')
        if dataset_num < 5:
            csv_path = os.path.join(base_path, f'dataset_{dataset_num}/matches_{dataset_num}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {dataset_titles[dataset_num]} with shape {df.shape}")
        else:
            csv_files = [os.path.join(base_path, f'dataset_5/matches_5_{j}.csv') for j in range(1, 11)]
            combined_df = pd.DataFrame()
            files_read = 0
            for file in csv_files:
                if not os.path.exists(file):
                    logger.warning(f"File not found: {file}")
                    continue
                try:
                    df = pd.read_csv(file)
                    df.columns = df.columns.str.lower().str.replace(' ', '_')
                    if not combined_df.empty:
                        for col in combined_df.columns:
                            if col not in df.columns:
                                df[col] = pd.NA
                        for col in df.columns:
                            if col not in combined_df.columns:
                                combined_df[col] = pd.NA
                    combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)
                    files_read += 1
                except Exception as e:
                    logger.error(f"Error reading {file}: {str(e)}")
            if files_read == 0:
                raise ValueError("No valid CSV files found for STOCKS")
            if combined_df.empty:
                raise ValueError("Combined STOCKS is empty after processing")
            df = combined_df
            logger.info(f"Loaded {dataset_titles[dataset_num]} with shape {df.shape}, {files_read} files combined")
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        cleaned_df = clean_data(df)
        logger.info(f"Cleaned {dataset_titles[dataset_num]} with shape {cleaned_df.shape}, columns: {cleaned_df.columns.tolist()}")
        return cleaned_df
    except Exception as e:
        logger.error(f"Error loading {dataset_titles[dataset_num]}: {str(e)}")
        return None

# Select numeric columns with sufficient unique values
def select_numeric_columns(df, num_cols):
    if not num_cols:
        logger.warning("No numeric columns available")
        return []
    valid_cols = [col for col in num_cols if df[col].nunique() > 1]
    logger.info(f"Numeric columns with >1 unique value: {valid_cols}")
    return valid_cols if valid_cols else num_cols[:1] if num_cols else []

# Select categorical columns
def select_categorical_columns(df, cat_cols):
    if not cat_cols:
        logger.warning("No categorical columns available")
        return []
    valid_cols = [col for col in cat_cols if df[col].nunique() > 0]
    logger.info(f"Categorical columns: {valid_cols}")
    return valid_cols

# Select a categorical pair for stacked bar
def select_categorical_pair(df, cat_cols):
    if len(cat_cols) < 2:
        logger.warning("Need at least two categorical columns for stacked bar")
        return None, None
    valid_pairs = []
    for i, col1 in enumerate(cat_cols):
        for col2 in cat_cols[i+1:]:
            ctab = pd.crosstab(df[col1], df[col2])
            if ctab.size > 0 and ctab.shape[0] > 1 and ctab.shape[1] > 1:
                try:
                    _, p, _, _ = chi2_contingency(ctab)
                    valid_pairs.append((col1, col2, p))
                except:
                    continue
    if valid_pairs:
        best_pair = min(valid_pairs, key=lambda x: x[2])
        logger.info(f"Selected categorical pair: {best_pair[0]}, {best_pair[1]}, p-value: {best_pair[2]}")
        return best_pair[0], best_pair[1]
    logger.info(f"No valid categorical pair, defaulting to: {cat_cols[0]}, {cat_cols[1]}")
    return cat_cols[0], cat_cols[1] if len(cat_cols) >= 2 else (None, None)

# Evaluate the best visualization type
def evaluate_best_viz(df):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_count = len(num_cols)
    cat_count = len(cat_cols)

    logger.info(f"Evaluating best viz: {num_count} numeric, {cat_count} categorical columns")

    if num_count >= 1 and cat_count >= 1:
        valid_num_cols = select_numeric_columns(df, num_cols)
        valid_cat_cols = select_categorical_columns(df, cat_cols)
        if valid_num_cols and valid_cat_cols and df[valid_cat_cols[0]].nunique() <= 20:
            logger.info("Selected box_plot for numeric and categorical columns")
            return 'box_plot'
    if num_count >= 2:
        valid_num_cols = select_numeric_columns(df, num_cols)
        if len(valid_num_cols) >= 2:
            logger.info("Selected heatmap for multiple numeric columns")
            return 'heatmap'
        logger.info("Selected scatter_plot for numeric pairs")
        return 'scatter_plot'
    elif num_count == 1:
        if df[num_cols[0]].nunique() > 1:
            logger.info("Selected histogram for single numeric column")
            return 'histogram'
    elif cat_count >= 2:
        cat_col1, cat_col2 = select_categorical_pair(df, cat_cols)
        if cat_col1 and cat_col2 and pd.crosstab(df[cat_col1], df[cat_col2]).size > 0:
            logger.info("Selected stacked_bar for categorical pair")
            return 'stacked_bar'
    elif cat_count >= 1:
        logger.info("Selected count_plot for categorical columns")
        return 'count_plot'
    logger.warning("No suitable columns for visualization")
    return None

# Create visualization
def create_visualization(df, label, viz_type):
    if not os.path.exists('static'):
        os.makedirs('static')

    img = BytesIO()
    try:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Creating {viz_type} for {label}, num_cols: {num_cols}, cat_cols: {cat_cols}")

        # Define a color palette for distinct colors
        palette = sns.color_palette("husl", n_colors=max(10, len(cat_cols) if cat_cols else 1))

        if viz_type == 'box_plot':
            valid_num_cols = select_numeric_columns(df, num_cols)
            valid_cat_cols = select_categorical_columns(df, cat_cols)
            valid_cat_cols = [col for col in valid_cat_cols if df[col].nunique() <= 20]
            if valid_num_cols and valid_cat_cols:
                n_plots = len(valid_num_cols) * len(valid_cat_cols[:1])
                cols = min(3, max(1, int(np.ceil(np.sqrt(n_plots)))))
                rows = int(np.ceil(n_plots / cols))
                fig = plt.figure(figsize=(12, 4 * rows))
                gs = gridspec.GridSpec(rows, cols)
                plot_idx = 0
                for num_col in valid_num_cols:
                    for cat_col in valid_cat_cols[:1]:
                        ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
                        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax, order=df[cat_col].value_counts().index[:10], 
                                    palette=palette[:df[cat_col].nunique()])
                        ax.set_title(f'{num_col} by {cat_col}', fontsize=10)
                        ax.set_xlabel(cat_col, fontsize=8)
                        ax.set_ylabel(num_col, fontsize=8)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        if df[cat_col].nunique() <= 10:
                            ax.legend(title=cat_col, labels=df[cat_col].value_counts().index[:10], fontsize=8, title_fontsize=10)
                        plot_idx += 1
                plt.suptitle(f'{label} - Distribution of Numeric Columns by Categories', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, "No suitable columns for box plot", ha='center', va='center', fontsize=12)
                plt.axis('off')
                logger.warning(f"No suitable columns for {label} box plot")

        elif viz_type == 'histogram':
            valid_num_cols = select_numeric_columns(df, num_cols)
            if valid_num_cols:
                n_cols = len(valid_num_cols)
                cols = min(3, max(1, int(np.ceil(np.sqrt(n_cols)))))
                rows = int(np.ceil(n_cols / cols))
                fig = plt.figure(figsize=(12, 4 * rows))
                gs = gridspec.GridSpec(rows, cols)
                for i, col in enumerate(valid_num_cols):
                    ax = fig.add_subplot(gs[i // cols, i % cols])
                    bins = min(50, max(10, int(np.sqrt(len(df)))))
                    sns.histplot(df[col], kde=True, bins=bins, ax=ax, color=palette[i % len(palette)], 
                                 kde_kws={'color': palette[(i + 1) % len(palette)], 'label': f'{col} KDE'})
                    ax.set_title(f'{col} (Unique: {df[col].nunique()})', fontsize=10)
                    ax.set_xlabel(col, fontsize=8)
                    ax.set_ylabel('Count', fontsize=8)
                    ax.legend(title='Data', labels=[f'{col} Histogram', f'{col} KDE'], fontsize=8, title_fontsize=10)
                plt.suptitle(f'{label} - Distributions of Numeric Columns', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, f"No suitable numeric columns for histogram\nFound: {num_cols}", 
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
                logger.warning(f"No suitable numeric columns for {label} histogram")

        elif viz_type == 'count_plot':
            valid_cat_cols = select_categorical_columns(df, cat_cols)
            if valid_cat_cols:
                n_cols = len(valid_cat_cols)
                cols = min(3, max(1, int(np.ceil(np.sqrt(n_cols)))))
                rows = int(np.ceil(n_cols / cols))
                fig = plt.figure(figsize=(12, 4 * rows))
                gs = gridspec.GridSpec(rows, cols)
                for i, col in enumerate(valid_cat_cols):
                    ax = fig.add_subplot(gs[i // cols, i % cols])
                    order = df[col].value_counts().index
                    sns.countplot(data=df, x=col, order=order, ax=ax, palette=palette[:df[col].nunique()])
                    ax.set_title(f'{col} (Unique: {df[col].nunique()})', fontsize=10)
                    ax.set_xlabel(col, fontsize=8)
                    ax.set_ylabel('Count', fontsize=8)
                    ax.tick_params(axis='x', rotation=45)
                    if df[col].nunique() <= 10:
                        ax.legend(title=col, labels=df[col].value_counts().index[:10], fontsize=8, title_fontsize=10)
                plt.suptitle(f'{label} - Frequencies of Categorical Columns', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, "No categorical columns for count plot", ha='center', va='center', fontsize=12)
                plt.axis('off')
                logger.warning(f"No suitable columns for {label} count plot")

        elif viz_type == 'scatter_plot':
            valid_num_cols = select_numeric_columns(df, num_cols)
            if len(valid_num_cols) >= 2:
                pairs = [(valid_num_cols[i], valid_num_cols[j]) for i in range(len(valid_num_cols)) 
                         for j in range(i+1, len(valid_num_cols))]
                n_pairs = len(pairs)
                cols = min(3, max(1, int(np.ceil(np.sqrt(n_pairs)))))
                rows = int(np.ceil(n_pairs / cols))
                fig = plt.figure(figsize=(12, 4 * rows))
                gs = gridspec.GridSpec(rows, cols)
                for i, (x_col, y_col) in enumerate(pairs):
                    ax = fig.add_subplot(gs[i // cols, i % cols])
                    marker_size = max(5, 50 / np.sqrt(len(df)))
                    if cat_cols:
                        sns.scatterplot(x=x_col, y=y_col, hue=cat_cols[0], data=df, alpha=0.6, s=marker_size, ax=ax, 
                                        palette=palette[:df[cat_cols[0]].nunique()])
                        ax.legend(title=cat_cols[0], fontsize=8, title_fontsize=10)
                    else:
                        sns.scatterplot(x=x_col, y=y_col, data=df, alpha=0.6, s=marker_size, ax=ax, 
                                        color=palette[i % len(palette)], label=f'{x_col} vs {y_col}')
                        ax.legend(fontsize=8, title_fontsize=10)
                    ax.set_title(f'{x_col} vs {y_col}', fontsize=10)
                    ax.set_xlabel(x_col, fontsize=8)
                    ax.set_ylabel(y_col, fontsize=8)
                plt.suptitle(f'{label} - Numeric Relationships', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, "Need at least two numeric columns for scatter plot", ha='center', va='center', fontsize=12)
                plt.axis('off')
                logger.warning(f"No suitable columns for {label} scatter plot")

        elif viz_type == 'heatmap':
            valid_num_cols = select_numeric_columns(df, num_cols)
            if len(valid_num_cols) >= 2:
                corr = df[valid_num_cols].corr()
                plt.figure(figsize=(max(8, len(valid_num_cols)), max(8, len(valid_num_cols))))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, annot_kws={'size': 8})
                plt.title(f'{label} - Correlation Heatmap of Numeric Columns', fontsize=14)
            else:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, "Need at least two numeric columns for heatmap", ha='center', va='center', fontsize=12)
                plt.axis('off')
                logger.warning(f"No suitable columns for {label} heatmap")

        elif viz_type == 'stacked_bar':
            valid_cat_cols = select_categorical_columns(df, cat_cols)
            cat_col1, cat_col2 = select_categorical_pair(df, valid_cat_cols)
            if cat_col1 and cat_col2 and df[cat_col1].nunique() > 0 and df[cat_col2].nunique() > 0:
                ctab = pd.crosstab(df[cat_col1], df[cat_col2])
                if not ctab.empty:
                    plt.figure(figsize=(12, 8))
                    ctab.plot(kind='bar', stacked=True, color=palette[:df[cat_col2].nunique()])
                    plt.title(f'{label} - Relationship between {cat_col1} and {cat_col2}', fontsize=14)
                    plt.xlabel(cat_col1, fontsize=12)
                    plt.ylabel('Count', fontsize=12)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.legend(title=cat_col2, fontsize=10, title_fontsize=12)
                else:
                    plt.figure(figsize=(12, 8))
                    plt.text(0.5, 0.5, "Crosstab empty, trying count plot", ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    logger.warning(f"Empty crosstab for {label} stacked bar, falling back to count plot")
                    if valid_cat_cols:
                        cat_col = valid_cat_cols[0]
                        plt.figure(figsize=(12, 8))
                        sns.countplot(data=df, x=cat_col, palette=palette[:df[cat_col].nunique()])
                        plt.title(f'{label} - Frequency of {cat_col}', fontsize=14)
                        plt.xlabel(cat_col, fontsize=12)
                        plt.ylabel('Count', fontsize=12)
                        plt.xticks(rotation=45, ha='right', fontsize=10)
                        if df[cat_col].nunique() <= 10:
                            plt.legend(title=cat_col, labels=df[cat_col].value_counts().index[:10], fontsize=10, title_fontsize=12)
            else:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, f"No suitable categorical columns for stacked bar\nCols: {cat_col1}, {cat_col2}", 
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
                logger.warning(f"No suitable columns for {label} stacked bar")

        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot = f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
        plt.close()
        logger.info(f"Generated {viz_type} for {label}")

    except Exception as e:
        logger.error(f"Error plotting {label}: {e}")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot = f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
        plt.close()

    return plot

@app.route('/')
def index():
    dataset_info = {}
    all_plots = {}
    dataset_tables = {}
    error = None

    try:
        for i in range(1, 6):
            label = dataset_titles[i]
            df = load_data(i)
            if df is None or df.empty:
                dataset_info[label] = {
                    'title': label,
                    'context': f"No data available for {label}"
                }
                all_plots[label] = None
                dataset_tables[label] = "<p class='text-gray-500 italic'>No data available</p>"
                logger.warning(f"No data for {label}")
                continue

            context = generate_csv_context(df)
            dataset_info[label] = {
                'title': label,
                'context': context
            }

            dataset_tables[label] = df.to_html(
                classes='w-full text-sm text-gray-700 border-collapse',
                index=False,
                border=0,
                escape=False
            )

            viz_type = evaluate_best_viz(df)
            if viz_type:
                plot = create_visualization(df, label, viz_type)
                all_plots[label] = [plot]
            else:
                all_plots[label] = None
                dataset_info[label]['context'] += "\nNo suitable visualization available."
                logger.warning(f"No suitable visualization for {label}")

        return render_template('index.html', info=dataset_info, plots=all_plots, tables=dataset_tables, error=error)

    except Exception as e:
        error = f"Error processing datasets: {str(e)}"
        logger.error(error)
        return render_template('index.html', info={}, plots={}, tables={}, error=error)

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=os.getenv('FLASK_ENV', 'development') == 'development')

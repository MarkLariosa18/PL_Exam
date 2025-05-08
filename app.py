import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template
import os
from io import BytesIO
import base64
import numpy as np
from scipy.stats import chi2_contingency
import threading
import time
import requests
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Dataset titles
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
def generate_context(df):
    try:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        context = []
        context.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
        if num_cols:
            context.append(f"Numeric columns: {', '.join(num_cols)}")
            for col in num_cols[:3]:
                context.append(f"- {col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, Min={df[col].min():.2f}, Max={df[col].max():.2f}")
        if cat_cols:
            context.append(f"Categorical columns: {', '.join(cat_cols)}")
            for col in cat_cols[:3]:
                top_cats = df[col].value_counts().index[:3]
                context.append(f"- {col}: Top categories: {', '.join(map(str, top_cats))}")
        return '\n'.join(context)
    except Exception as e:
        return f"Error generating context: {str(e)}"

# Load data for a dataset
def load_data(dataset_num):
    try:
        if dataset_num < 5:
            csv_path = f'dataset_{dataset_num}/matches_{dataset_num}.csv'
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            csv_files = [f'dataset_5/matches_5_{j}.csv' for j in range(1, 11)]
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
                raise ValueError("No valid CSV files found for Dataset 5")
            if combined_df.empty:
                raise ValueError("Combined Dataset 5 is empty after processing")
            df = combined_df
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return clean_data(df)
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_num}: {str(e)}")
        return None

# Select top numeric columns based on variance
def select_numeric_columns(df, num_cols, max_cols=3):
    if not num_cols:
        return []
    variances = df[num_cols].var()
    sorted_cols = variances.sort_values(ascending=False).index.tolist()
    return sorted_cols[:min(max_cols, len(sorted_cols))]

# Select top categorical columns based on value counts
def select_categorical_columns(df, cat_cols, max_cols=3):
    if not cat_cols:
        return []
    value_counts = [df[col].value_counts().iloc[:10].sum() for col in cat_cols]
    sorted_cols = [cat_cols[i] for i in np.argsort(value_counts)[::-1]]
    return sorted_cols[:min(max_cols, len(sorted_cols))]

# Select the most correlated numeric pair
def select_numeric_pair(df, num_cols):
    if len(num_cols) < 2:
        return None, None, 0
    corr = df[num_cols].corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_corr = corr.max().max()
    if max_corr > 0:
        pair = corr.stack().idxmax()
        return pair[0], pair[1], max_corr
    return num_cols[0], num_cols[1], 0

# Select the most associated categorical pair (chi-squared test)
def select_categorical_pair(df, cat_cols):
    if len(cat_cols) < 2:
        return None, None, 1
    best_pair = None
    min_p = 1
    for i, col1 in enumerate(cat_cols):
        for col2 in cat_cols[i+1:]:
            ctab = pd.crosstab(df[col1], df[col2])
            if ctab.size > 0 and ctab.shape[0] > 1 and ctab.shape[1] > 1:
                try:
                    _, p, _, _ = chi2_contingency(ctab)
                    if p < min_p:
                        min_p = p
                        best_pair = (col1, col2)
                except:
                    continue
    return best_pair if best_pair else (cat_cols[0], cat_cols[1]), min_p

# Dataset-specific column selection
def select_dataset_columns(df, dataset_num):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if dataset_num == 1:
        football_num = ['goals', 'assists', 'shots', 'passes', 'tackles', 'saves', 'minutes_played']
        selected_num = [col for col in football_num if col in num_cols][:3]
        if len(selected_num) < 3:
            selected_num.extend(select_numeric_columns(df, [col for col in num_cols if col not in selected_num], max_cols=3-len(selected_num)))
        selected_cat = select_categorical_columns(df, cat_cols, max_cols=2)
    elif dataset_num == 3:
        selected_num = ['carat'] if 'carat' in num_cols else select_numeric_columns(df, num_cols, max_cols=1)
        selected_cat = [col for col in ['cut', 'color'] if col in cat_cols]
        if len(selected_cat) < 2:
            selected_cat.extend(select_categorical_columns(df, [col for col in cat_cols if col not in selected_cat], max_cols=2-len(selected_cat)))
    elif dataset_num == 4:
        selected_num = [col for col in ['distinct_called_numbers', 'frequency_of_use'] if col in num_cols]
        if len(selected_num) < 2:
            selected_num.extend(select_numeric_columns(df, [col for col in num_cols if col not in selected_num], max_cols=2-len(selected_num)))
        selected_cat = ['age_group'] if 'age_group' in cat_cols else select_categorical_columns(df, cat_cols, max_cols=1)
    else:
        selected_num = select_numeric_columns(df, num_cols, max_cols=3)
        selected_cat = select_categorical_columns(df, cat_cols, max_cols=3)
    
    return selected_num, selected_cat

# Evaluate visualization options and score them
def evaluate_visualizations(df, dataset_num):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    scores = {}

    selected_num, selected_cat = select_dataset_columns(df, dataset_num)

    if dataset_num == 5:
        if selected_num:
            variance = df[selected_num].var().mean() if selected_num else 0
            scores['histogram'] = variance / (df[selected_num].std().mean() + 1) if variance > 0 else 0
        else:
            scores['histogram'] = 0

        if selected_cat:
            count_score = sum(df[col].value_counts().iloc[:10].sum() for col in selected_cat) / len(df) / max(1, len(selected_cat))
            scores['count_plot'] = count_score if max(df[col].nunique() for col in selected_cat) <= 20 else count_score * 0.5
        else:
            scores['count_plot'] = 0

        if len(num_cols) >= 2:
            x_col, y_col, corr = select_numeric_pair(df, num_cols)
            scores['scatter_plot'] = corr if corr > 0.1 else corr * 0.5
        else:
            scores['scatter_plot'] = 0

        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            scores['heatmap'] = corr_matrix.mean().mean() if not corr_matrix.empty else 0
        else:
            scores['heatmap'] = 0

        if len(cat_cols) >= 2:
            (cat_col1, cat_col2), p = select_categorical_pair(df, cat_cols)
            assoc_score = 1 - p if p < 1 else 0
            nunique1 = df[cat_col1].nunique() if cat_col1 else float('inf')
            nunique2 = df[cat_col2].nunique() if cat_col2 else float('inf')
            scores['stacked_bar'] = assoc_score if nunique1 <= 20 and nunique2 <= 20 else assoc_score * 0.5
        else:
            scores['stacked_bar'] = 0
    else:
        if selected_num:
            iqr = (df[selected_num].quantile(0.75) - df[selected_num].quantile(0.25)).mean() if selected_num else 0
            scores['box_plot'] = iqr / (df[selected_num].std().mean() + 1) if iqr > 0 else 0
        else:
            scores['box_plot'] = 0

        if selected_cat:
            count_score = sum(df[col].value_counts().iloc[:10].sum() for col in selected_cat) / len(df) / max(1, len(selected_cat))
            scores['bar_plot'] = count_score if max(df[col].nunique() for col in selected_cat) <= 20 else count_score * 0.5
        else:
            scores['bar_plot'] = 0

        if len(num_cols) >= 2:
            x_col, y_col, corr = select_numeric_pair(df, num_cols)
            scores['line_plot'] = corr if corr > 0.1 else corr * 0.5
        else:
            scores['line_plot'] = 0

        if selected_num:
            variance = df[selected_num].var().mean() if selected_num else 0
            scores['violin_plot'] = variance / (df[selected_num].std().mean() + 1) if variance > 0 else 0
        else:
            scores['violin_plot'] = 0

        if len(cat_cols) >= 2:
            (cat_col1, cat_col2), p = select_categorical_pair(df, cat_cols)
            assoc_score = 1 - p if p < 1 else 0
            nunique1 = df[cat_col1].nunique() if cat_col1 else float('inf')
            nunique2 = df[cat_col2].nunique() if cat_col2 else float('inf')
            scores['clustered_bar'] = assoc_score if nunique1 <= 20 and nunique2 <= 20 else assoc_score * 0.5
        else:
            scores['clustered_bar'] = 0

    return scores

# Create visualization based on selected type
def create_visualization(df, title, viz_type, dataset_num):
    if not os.path.exists('static'):
        os.makedirs('static')

    img = BytesIO()
    fig = plt.figure(figsize=(12, 8))  # Explicit figure creation
    try:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        selected_num, selected_cat = select_dataset_columns(df, dataset_num)

        if dataset_num == 5:
            if viz_type == 'histogram':
                if selected_num:
                    for col in selected_num:
                        sns.histplot(df[col], kde=True, bins=30, label=col, alpha=0.5)
                    plt.title(f'{title} - Distribution of Numeric Columns', fontsize=14, pad=15)
                    plt.xlabel('Value', fontsize=12)
                    plt.ylabel('Count', fontsize=12)
                    plt.legend(title='Column', fontsize=10, title_fontsize=12)
                else:
                    plt.text(0.5, 0.5, "No numeric columns for histogram", ha='center', va='center', fontsize=12)

            elif viz_type == 'count_plot':
                if selected_cat:
                    fig, axes = plt.subplots(1, len(selected_cat), figsize=(12, 8), sharey=True)
                    axes = [axes] if len(selected_cat) == 1 else axes
                    for ax, col in zip(axes, selected_cat):
                        top_categories = df[col].value_counts().index[:10]
                        sns.countplot(data=df[df[col].isin(top_categories)], x=col, order=top_categories, ax=ax)
                        ax.set_title(f'{col}', fontsize=12)
                        ax.set_xlabel(col, fontsize=10)
                        ax.set_ylabel('Count', fontsize=10)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                    plt.suptitle(f'{title} - Frequency of Categorical Columns', fontsize=14, y=1.05)
                    plt.tight_layout()
                else:
                    plt.text(0.5, 0.5, "No categorical columns for count plot", ha='center', va='center', fontsize=12)

            elif viz_type == 'scatter_plot':
                x_col, y_col, _ = select_numeric_pair(df, num_cols)
                hue_col = selected_cat[0] if selected_cat else None
                if x_col and y_col:
                    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df.sample(n=min(1000, len(df)), random_state=1), alpha=0.6)
                    plt.title(f'{title} - Relationship between {x_col} and {y_col}', fontsize=14, pad=15)
                    plt.xlabel(x_col, fontsize=12)
                    plt.ylabel(y_col, fontsize=12)
                    if hue_col:
                        plt.legend(title=hue_col, fontsize=10, title_fontsize=12)
                else:
                    plt.text(0.5, 0.5, "Need at least two numeric columns for scatter plot", ha='center', va='center', fontsize=12)

            elif viz_type == 'heatmap':
                if len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, annot_kws={'size': 10})
                    plt.title(f'{title} - Correlation Heatmap of Numeric Variables', fontsize=14, pad=15)
                else:
                    plt.text(0.5, 0.5, "Need at least two numeric columns for heatmap", ha='center', va='center', fontsize=12)

            elif viz_type == 'stacked_bar':
                cat_col1, cat_col2 = select_categorical_pair(df, cat_cols)[0]
                if cat_col1 and cat_col2 and df[cat_col1].nunique() <= 10 and df[cat_col2].nunique() <= 10:
                    top_cat1 = df[cat_col1].value_counts().index[:10]
                    top_cat2 = df[cat_col2].value_counts().index[:10]
                    filtered_df = df[df[cat_col1].isin(top_cat1) & df[cat_col2].isin(top_cat2)]
                    if not filtered_df.empty:
                        ctab = pd.crosstab(filtered_df[cat_col1], filtered_df[cat_col2])
                        ctab.plot(kind='bar', stacked=True)
                        plt.title(f'{title} - Relationship between {cat_col1} and {cat_col2}', fontsize=14, pad=15)
                        plt.xlabel(cat_col1, fontsize=12)
                        plt.ylabel('Count', fontsize=12)
                        plt.xticks(rotation=45, ha='right', fontsize=10)
                        plt.legend(title=cat_col2, fontsize=10, title_fontsize=12)
                    else:
                        plt.text(0.5, 0.5, "No data available after filtering categories", ha='center', va='center', fontsize=12)
                else:
                    plt.text(0.5, 0.5, f"Need two categorical columns with ≤10 unique values\nFound: {len(cat_cols)} columns, "
                                       f"{df[cat_col1].nunique() if cat_col1 else 0} and "
                                       f"{df[cat_col2].nunique() if cat_col2 else 0} unique values",
                             ha='center', va='center', fontsize=12)
        else:
            if viz_type == 'box_plot':
                if selected_num:
                    sns.boxplot(data=df[selected_num])
                    plt.title(f'{title} - Box Plot of Numeric Columns', fontsize=14, pad=15)
                    plt.ylabel('Value', fontsize=12)
                    plt.legend(labels=selected_num, title='Column', fontsize=10, title_fontsize=12)
                else:
                    plt.text(0.5, 0.5, "No numeric columns for box plot", ha='center', va='center', fontsize=12)

            elif viz_type == 'bar_plot':
                if selected_cat:
                    fig, axes = plt.subplots(1, len(selected_cat), figsize=(12, 8), sharey=True)
                    axes = [axes] if len(selected_cat) == 1 else axes
                    for ax, col in zip(axes, selected_cat):
                        top_categories = df[col].value_counts().index[:10]
                        sns.barplot(x=df[col].value_counts().index[:10], y=df[col].value_counts().values[:10], ax=ax)
                        ax.set_title(f'{col}', fontsize=12)
                        ax.set_xlabel(col, fontsize=10)
                        ax.set_ylabel('Count', fontsize=10)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                    plt.suptitle(f'{title} - Bar Plot of Categorical Columns', fontsize=14, y=1.05)
                    plt.tight_layout()
                else:
                    plt.text(0.5, 0.5, "No categorical columns for bar plot", ha='center', va='center', fontsize=12)

            elif viz_type == 'line_plot':
                x_col, y_col, _ = select_numeric_pair(df, num_cols)
                hue_col = selected_cat[0] if selected_cat else None
                if x_col and y_col:
                    sample_df = df.sample(n=min(1000, len(df)), random_state=1).sort_values(x_col)
                    sns.lineplot(x=sample_df[x_col], y=sample_df[y_col], hue=hue_col)
                    plt.title(f'{title} - Line Plot of {x_col} vs {y_col}', fontsize=14, pad=15)
                    plt.xlabel(x_col, fontsize=12)
                    plt.ylabel(y_col, fontsize=12)
                    if hue_col:
                        plt.legend(title=hue_col, fontsize=10, title_fontsize=12)
                else:
                    plt.text(0.5, 0.5, "Need at least two numeric columns for line plot", ha='center', va='center', fontsize=12)

            elif viz_type == 'violin_plot':
                if selected_num:
                    sns.violinplot(data=df[selected_num])
                    plt.title(f'{title} - Violin Plot of Numeric Columns', fontsize=14, pad=15)
                    plt.ylabel('Value', fontsize=12)
                    plt.legend(labels=selected_num, title='Column', fontsize=10, title_fontsize=12)
                else:
                    plt.text(0.5, 0.5, "No numeric columns for violin plot", ha='center', va='center', fontsize=12)

            elif viz_type == 'clustered_bar':
                cat_col1, cat_col2 = select_categorical_pair(df, cat_cols)[0]
                if cat_col1 and cat_col2 and df[cat_col1].nunique() <= 10 and df[cat_col2].nunique() <= 10:
                    top_cat1 = df[cat_col1].value_counts().index[:10]
                    top_cat2 = df[cat_col2].value_counts().index[:10]
                    filtered_df = df[df[cat_col1].isin(top_cat1) & df[cat_col2].isin(top_cat2)]
                    if not filtered_df.empty:
                        sns.catplot(data=filtered_df, x=cat_col1, hue=cat_col2, kind='count', height=8, aspect=1.5)
                        plt.title(f'{title} - Clustered Bar Plot of {cat_col1} and {cat_col2}', fontsize=14, pad=15)
                        plt.xlabel(cat_col1, fontsize=12)
                        plt.ylabel('Count', fontsize=12)
                        plt.xticks(rotation=45, ha='right', fontsize=10)
                        plt.legend(title=cat_col2, fontsize=10, title_fontsize=12)
                    else:
                        plt.text(0.5, 0.5, "No data available after filtering categories", ha='center', va='center', fontsize=12)
                else:
                    plt.text(0.5, 0.5, f"Need two categorical columns with ≤10 unique values\nFound: {len(cat_cols)} columns, "
                                       f"{df[cat_col1].nunique() if cat_col1 else 0} and "
                                       f"{df[cat_col2].nunique() if cat_col2 else 0} unique values",
                             ha='center', va='center', fontsize=12)

        plt.tight_layout(pad=2.0)
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot = f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
    except Exception as e:
        logger.error(f"Error plotting {title}: {e}")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", ha='center', va='center', fontsize=12)
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot = f'data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}'
    finally:
        plt.close(fig)  # Ensure figure is closed

    return plot

# Assign unique visualizations to each dataset
def assign_visualizations(datasets_scores):
    available_viz_dataset_5 = ['histogram', 'count_plot', 'scatter_plot', 'heatmap', 'stacked_bar']
    available_viz_others = ['box_plot', 'bar_plot', 'line_plot', 'violin_plot', 'clustered_bar']
    assignments = {}
    used_viz = set()

    for dataset_num in range(1, 6):
        label = f'Dataset {dataset_num}'
        scores = datasets_scores.get(label, {})
        if not scores:
            assignments[label] = None
            continue
        available_viz = available_viz_dataset_5 if dataset_num == 5 else available_viz_others
        viz_options = [(viz, score) for viz, score in scores.items() if viz in available_viz and viz not in used_viz]
        if not viz_options:
            viz_options = [(viz, 0) for viz in available_viz if viz not in used_viz]
        if viz_options:
            best_viz = max(viz_options, key=lambda x: x[1])[0]
            assignments[label] = best_viz
            used_viz.add(best_viz)
        else:
            assignments[label] = None
    return assignments

@app.route('/')
def index():
    dataset_info = {}
    all_plots = {}
    error = None
    datasets_scores = {}

    try:
        for i in range(1, 6):
            label = f'Dataset {i}'
            title = dataset_titles[i]

            df = load_data(i)
            if df is None or df.empty:
                dataset_info[label] = {'context': f"No data available for {title}", 'title': title}
                all_plots[label] = None
                continue

            context = generate_context(df)
            dataset_info[label] = {'context': context, 'title': title}

            datasets_scores[label] = evaluate_visualizations(df, i)

        viz_assignments = assign_visualizations(datasets_scores)

        for i in range(1, 6):
            label = f'Dataset {i}'
            title = dataset_titles[i]
            if label not in dataset_info:
                continue
            df = load_data(i)
            if df is None or df.empty:
                continue
            viz_type = viz_assignments.get(label)
            if viz_type:
                all_plots[label] = create_visualization(df, title, viz_type, i)
            else:
                all_plots[label] = None
                dataset_info[label]['context'] += "\nNo suitable visualization available."

        return render_template('index.html', info=dataset_info, plots=all_plots, error=error)

    except Exception as e:
        logger.error(f"Error processing datasets: {str(e)}")
        error = f"Error processing datasets: {str(e)}"
        return render_template('index.html', info={}, plots={}, error=error)

@app.route('/health')
def health():
    return 'OK', 200

# Ping function to keep the app alive
def keep_alive():
    url = os.getenv('RENDER_EXTERNAL_URL')
    if not url:
        logger.warning("No RENDER_EXTERNAL_URL set, skipping ping")
        return
    ping_url = urljoin(url, '/')
    while True:
        try:
            response = requests.get(ping_url, timeout=10)
            logger.info(f"Pinged {ping_url} - Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error pinging {ping_url}: {str(e)}")
        time.sleep(14 * 60)  # Sleep for 14 minutes

# Start the keep-alive thread
if os.getenv('RENDER'):
    threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Gunicorn on 0.0.0.0:{port}")
    os.system(f"gunicorn -w 2 --threads 2 --preload -b 0.0.0.0:{port} app:app")
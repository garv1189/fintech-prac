import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_plotly_events import plotly_events

# Set random seed for reproducibility
random.seed(42)
Faker.seed(42)
fake = Faker()


def generate_esg_data(num_companies=100):
    """Generate fake ESG data for companies"""
    sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods',
               'Industrial', 'Materials', 'Utilities', 'Real Estate', 'Telecommunications']

    data = []
    years = list(range(2019, 2024))

    for _ in range(num_companies):
        company = fake.company()
        sector = random.choice(sectors)
        market_cap = random.uniform(1e9, 1e12)  # $1B to $1T

        # Generate consistent but slightly varying scores over years
        base_environmental = random.uniform(50, 95)
        base_social = random.uniform(50, 95)
        base_governance = random.uniform(50, 95)

        for year in years:
            # Add some random variation year over year
            environmental = min(100, max(0, base_environmental + random.uniform(-5, 5)))
            social = min(100, max(0, base_social + random.uniform(-5, 5)))
            governance = min(100, max(0, base_governance + random.uniform(-5, 5)))

            # Calculate composite score
            esg_score = (environmental + social + governance) / 3

            data.append({
                'Year': year,
                'Company': company,
                'Sector': sector,
                'Market_Cap': market_cap,
                'Environmental_Score': environmental,
                'Social_Score': social,
                'Governance_Score': governance,
                'ESG_Score': esg_score,
                'Carbon_Emissions': random.uniform(100000, 1000000),
                'Water_Usage': random.uniform(1000000, 10000000),
                'Renewable_Energy_Percentage': random.uniform(0, 100),
                'Employee_Satisfaction': random.uniform(60, 95),
                'Diversity_Score': random.uniform(50, 90),
                'Board_Independence': random.uniform(60, 95)
            })

    return pd.DataFrame(data)


def create_sector_comparison(df, year):
    """Create sector comparison visualization"""
    sector_avg = df[df['Year'] == year].groupby('Sector')[
        ['Environmental_Score', 'Social_Score', 'Governance_Score']].mean().reset_index()

    fig = go.Figure()

    categories = ['Environmental_Score', 'Social_Score', 'Governance_Score']

    for sector in sector_avg['Sector']:
        sector_data = sector_avg[sector_avg['Sector'] == sector]
        fig.add_trace(go.Scatterpolar(
            r=[sector_data[cat].iloc[0] for cat in categories],
            theta=categories,
            fill='toself',
            name=sector
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title=f"ESG Scores by Sector ({year})"
    )

    return fig


def create_trend_analysis(df, selected_companies):
    """Create trend analysis visualization"""
    trend_data = df[df['Company'].isin(selected_companies)]

    fig = go.Figure()

    for company in selected_companies:
        company_data = trend_data[trend_data['Company'] == company]
        fig.add_trace(go.Scatter(
            x=company_data['Year'],
            y=company_data['ESG_Score'],
            name=company,
            mode='lines+markers'
        ))

    fig.update_layout(
        title="ESG Score Trends Over Time",
        xaxis_title="Year",
        yaxis_title="ESG Score",
        yaxis_range=[0, 100]
    )

    return fig


def calculate_statistical_metrics(df, company=None):
    """Calculate statistical metrics for ESG scores"""
    if company:
        data = df[df['Company'] == company]
    else:
        data = df

    metrics = {}
    for score_type in ['Environmental_Score', 'Social_Score', 'Governance_Score', 'ESG_Score']:
        metrics[score_type] = {
            'Mean': data[score_type].mean(),
            'Median': data[score_type].median(),
            'Std Dev': data[score_type].std(),
            'YoY Change': (data[data['Year'] == data['Year'].max()][score_type].mean() /
                           data[data['Year'] == data['Year'].min()][score_type].mean() - 1) * 100,
            'Percentile Rank': stats.percentileofscore(df[score_type], data[score_type].mean())
        }

    return metrics


def perform_cluster_analysis(df, year):
    """Perform K-means clustering on ESG data"""
    df_year = df[df['Year'] == year].copy()

    features = ['Environmental_Score', 'Social_Score', 'Governance_Score']
    X = df_year[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df_year['Cluster'] = kmeans.fit_predict(X_scaled)

    return df_year, kmeans.cluster_centers_


def create_correlation_heatmap(df, year):
    """Create correlation heatmap for ESG metrics"""
    df_year = df[df['Year'] == year]
    metrics = ['Environmental_Score', 'Social_Score', 'Governance_Score',
               'Carbon_Emissions', 'Water_Usage', 'Renewable_Energy_Percentage',
               'Employee_Satisfaction', 'Diversity_Score', 'Board_Independence']

    corr_matrix = df_year[metrics].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=metrics,
        y=metrics,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title=f"Correlation Heatmap ({year})",
        xaxis_tickangle=-45
    )

    return fig


def train_prophet_model(df, company, metric):
    """Train Facebook Prophet model for ESG predictions"""
    company_data = df[df['Company'] == company][[metric, 'Year']].copy()
    company_data.columns = ['y', 'ds']
    company_data['ds'] = pd.to_datetime(company_data['ds'].astype(str))

    model = Prophet(yearly_seasonality=True)
    model.fit(company_data)

    future_dates = model.make_future_dataframe(periods=3, freq='Y')
    forecast = model.predict(future_dates)

    return forecast


def calculate_cagr(data):
    """Calculate Compound Annual Growth Rate"""
    years = data['Year'].unique()
    if len(years) < 2:
        return 0

    start_value = data[data['Year'] == min(years)]['ESG_Score'].mean()
    end_value = data[data['Year'] == max(years)]['ESG_Score'].mean()
    num_years = max(years) - min(years)

    if start_value <= 0:
        return 0

    return (((end_value / start_value) ** (1 / num_years)) - 1) * 100


def calculate_volatility(data):
    """Calculate year-over-year volatility"""
    return data.groupby('Year')['ESG_Score'].mean().pct_change().std() * 100


def calculate_trend_strength(data):
    """Calculate trend strength using R-squared of linear regression"""
    years = data['Year'].unique()
    if len(years) < 2:
        return 0

    yearly_means = data.groupby('Year')['ESG_Score'].mean().reset_index()
    X = yearly_means['Year'].values.reshape(-1, 1)
    y = yearly_means['ESG_Score'].values

    model = LinearRegression()
    model.fit(X, y)
    return model.score(X, y) * 100


def calculate_advanced_statistics(df, company=None):
    """Calculate advanced statistical metrics"""
    if company:
        data = df[df['Company'] == company]
    else:
        data = df

    stats_dict = {
        'Basic Statistics': {
            'Mean': data['ESG_Score'].mean(),
            'Median': data['ESG_Score'].median(),
            'Std Dev': data['ESG_Score'].std(),
            'Skewness': data['ESG_Score'].skew(),
            'Kurtosis': data['ESG_Score'].kurtosis()
        },
        'Percentiles': {
            '25th': data['ESG_Score'].quantile(0.25),
            '50th': data['ESG_Score'].quantile(0.50),
            '75th': data['ESG_Score'].quantile(0.75),
            '90th': data['ESG_Score'].quantile(0.90)
        },
        'Year-over-Year': {
            'CAGR': calculate_cagr(data),
            'Volatility': calculate_volatility(data),
            'Trend Strength': calculate_trend_strength(data)
        }
    }

    return stats_dict


def generate_pdf_report(df, company, stats, predictions):
    """Generate PDF report for company ESG analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph(f"ESG Analysis Report - {company}", styles['Title']))

    # Company Overview
    company_data = df[df['Company'] == company].iloc[0]
    overview = [
        ['Metric', 'Value'],
        ['Sector', company_data['Sector']],
        ['ESG Score', f"{company_data['ESG_Score']:.2f}"],
        ['Environmental Score', f"{company_data['Environmental_Score']:.2f}"],
        ['Social Score', f"{company_data['Social_Score']:.2f}"],
        ['Governance Score', f"{company_data['Governance_Score']:.2f}"]
    ]

    elements.append(Paragraph("Company Overview", styles['Heading1']))
    elements.append(Table(overview))

    # Statistics
    elements.append(Paragraph("Statistical Analysis", styles['Heading1']))
    stats_table = [[k, f"{v:.2f}"] for k, v in stats['Basic Statistics'].items()]
    elements.append(Table(stats_table))

    # Predictions
    elements.append(Paragraph("Future Predictions", styles['Heading1']))
    pred_table = [['Year', 'Predicted ESG Score']]
    for year, pred in predictions.items():
        pred_table.append([str(year), f"{pred:.2f}"])
    elements.append(Table(pred_table))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()

    return pdf


def create_interactive_scatter(df, year):
    """Create interactive scatter plot with brushing and linking"""
    fig = px.scatter(df[df['Year'] == year],
                     x='Environmental_Score',
                     y='Social_Score',
                     size='Market_Cap',
                     color='Sector',
                     hover_data=['Company', 'Governance_Score'],
                     custom_data=['Company'])

    fig.update_layout(dragmode='select')
    return fig


def export_to_excel(df, filename):
    """Export analysis to Excel with formatting"""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # Write main data
    df.to_excel(writer, sheet_name='ESG Data', index=False)

    # Add summary sheet
    summary = pd.DataFrame({
        'Metric': ['Average ESG Score', 'Top Performer', 'Most Improved'],
        'Value': [
            df['ESG_Score'].mean(),
            df.loc[df['ESG_Score'].idxmax(), 'Company'],
            calculate_most_improved(df)
        ]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

    writer.close()
    return output.getvalue()


def calculate_most_improved(df):
    """Calculate the most improved company based on ESG score"""
    company_improvement = df.groupby('Company').agg({
        'ESG_Score': lambda x: x.iloc[-1] - x.iloc[0]
    })
    return company_improvement['ESG_Score'].idxmax()


def create_radar_comparison(df):
    """Create radar chart comparison for selected companies"""
    metrics = ['Environmental_Score', 'Social_Score', 'Governance_Score']

    fig = go.Figure()

    for company in df['Company'].unique():
        company_data = df[df['Company'] == company].iloc[0]
        fig.add_trace(go.Scatterpolar(
            r=[company_data[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=company
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="ESG Components Comparison"
    )

    return fig

def generate_interactive_dashboard(df):
    """Generate interactive HTML dashboard"""
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESG Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .dashboard-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .chart-container { margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div id="overview" class="chart-container"></div>
            <div id="trends" class="chart-container"></div>
            <div id="sector" class="chart-container"></div>
        </div>
    </body>
    </html>
    """

    # Create visualizations
    overview_fig = px.box(df, x='Sector', y='ESG_Score', title="ESG Scores by Sector")

    trends_fig = px.line(
        df.groupby(['Year', 'Sector'])['ESG_Score'].mean().reset_index(),
        x='Year', y='ESG_Score', color='Sector',
        title="ESG Score Trends by Sector"
    )

    sector_fig = px.scatter(
        df,
        x='Environmental_Score',
        y='Social_Score',
        color='Sector',
        size='Market_Cap',
        hover_data=['Company', 'Governance_Score'],
        title="Environmental vs Social Scores by Sector"
    )

    # Combine visualizations into the template
    dashboard_html = template.replace(
        '<div id="overview" class="chart-container"></div>',
        f'<div id="overview" class="chart-container">{overview_fig.to_html(full_html=False)}</div>'
    ).replace(
        '<div id="trends" class="chart-container"></div>',
        f'<div id="trends" class="chart-container">{trends_fig.to_html(full_html=False)}</div>'
    ).replace(
        '<div id="sector" class="chart-container"></div>',
        f'<div id="sector" class="chart-container">{sector_fig.to_html(full_html=False)}</div>'
    )

    return dashboard_html


def main():
    st.set_page_config(layout="wide")
    st.title("Advanced ESG Analysis Dashboard with ML Predictions")

    # Generate initial data
    df = generate_esg_data(100)

    # Add year selection at the top level
    selected_year = st.sidebar.selectbox("Select Year", sorted(df['Year'].unique()))

    # Create tabs
    tabs = st.tabs(["Overview", "Statistical Analysis", "ML Predictions", "Interactive Analysis", "Reports"])

    with tabs[0]:
        st.subheader("ESG Overview")
        col1, col2 = st.columns(2)

        with col1:
            sector_fig = create_sector_comparison(df, selected_year)
            st.plotly_chart(sector_fig, use_container_width=True)

        with col2:
            selected_companies = st.multiselect(
                "Select Companies for Trend Analysis",
                options=sorted(df['Company'].unique()),
                default=sorted(df['Company'].unique())[:3]
            )
            if selected_companies:
                trend_fig = create_trend_analysis(df, selected_companies)
                st.plotly_chart(trend_fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Statistical Analysis")

        col1, col2 = st.columns([1, 2])

        with col1:
            selected_company = st.selectbox(
                "Select Company for Analysis",
                options=sorted(df['Company'].unique())
            )

            stats = calculate_statistical_metrics(df, selected_company)
            st.write("ESG Score Statistics:", stats)

        with col2:
            corr_fig = create_correlation_heatmap(df, selected_year)
            st.plotly_chart(corr_fig, use_container_width=True)

        # Add cluster analysis
        st.subheader("Cluster Analysis")
        clustered_df, centers = perform_cluster_analysis(df, selected_year)

        cluster_fig = px.scatter(
            clustered_df,
            x='Environmental_Score',
            y='Social_Score',
            color='Cluster',
            hover_data=['Company', 'Sector']
        )
        st.plotly_chart(cluster_fig, use_container_width=True)

    with tabs[2]:
        st.subheader("ESG Score Predictions")

        col1, col2 = st.columns([1, 2])

        with col1:
            pred_company = st.selectbox(
                "Select Company for Predictions",
                options=sorted(df['Company'].unique()),
                key="pred_company"
            )

            metric = st.selectbox(
                "Select Metric to Predict",
                options=['ESG_Score', 'Environmental_Score', 'Social_Score', 'Governance_Score']
            )

            advanced_stats = calculate_advanced_statistics(df, pred_company)
            st.write("Advanced Statistics:", advanced_stats)

        with col2:
            forecast = train_prophet_model(df, pred_company, metric)

            fig = go.Figure()

            # Historical data
            hist_data = df[df['Company'] == pred_company]
            fig.add_trace(go.Scatter(
                x=hist_data['Year'],
                y=hist_data[metric],
                name='Historical',
                mode='markers+lines'
            ))

            # Predictions
            fig.add_trace(go.Scatter(
                x=forecast['ds'].dt.year[-3:],
                y=forecast['yhat'][-3:],
                name='Predicted',
                mode='markers+lines',
                line=dict(dash='dot')
            ))

            fig.update_layout(
                title=f"{metric} Predictions for {pred_company}",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show prediction intervals
            pred_df = pd.DataFrame({
                'Year': forecast['ds'].dt.year[-3:],
                'Predicted': forecast['yhat'][-3:].round(2),
                'Lower Bound': forecast['yhat_lower'][-3:].round(2),
                'Upper Bound': forecast['yhat_upper'][-3:].round(2)
            })
            st.dataframe(pred_df)

    with tabs[3]:
        st.subheader("Interactive Analysis")

        scatter_fig = create_interactive_scatter(df, selected_year)
        selected_points = plotly_events(scatter_fig)

        if selected_points:
            selected_companies = [point['customdata'][0] for point in selected_points]
            st.write("Selected Companies Analysis:")
            selected_df = df[df['Company'].isin(selected_companies)]

            col1, col2 = st.columns(2)

            with col1:
                comparison_fig = px.bar(
                    selected_df,
                    x='Company',
                    y=['Environmental_Score', 'Social_Score', 'Governance_Score'],
                    title="ESG Component Comparison"
                )
                st.plotly_chart(comparison_fig)

            with col2:
                radar_fig = create_radar_comparison(selected_df)
                st.plotly_chart(radar_fig)

    with tabs[4]:
        st.subheader("Export Reports")

        report_type = st.selectbox(
            "Select Report Type",
            options=['PDF Report', 'Excel Export', 'Interactive Dashboard']
        )

        if report_type == 'PDF Report':
            report_company = st.selectbox(
                "Select Company for Report",
                options=sorted(df['Company'].unique()),
                key="report_company"
            )

            stats = calculate_advanced_statistics(df, report_company)
            forecast = train_prophet_model(df, report_company, 'ESG_Score')
            predictions = dict(zip(forecast['ds'].dt.year[-3:], forecast['yhat'][-3:]))

            pdf = generate_pdf_report(df, report_company, stats, predictions)

            st.download_button(
                "Download PDF Report",
                pdf,
                f"ESG_Report_{report_company}.pdf",
                "application/pdf"
            )

        elif report_type == 'Excel Export':
            excel_file = export_to_excel(df, "ESG_Analysis.xlsx")

            st.download_button(
                "Download Excel Report",
                excel_file,
                "ESG_Analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.write("Interactive dashboard will open in a new tab")
            dashboard_html = generate_interactive_dashboard(df)

            st.download_button(
                "Download Interactive Dashboard",
                dashboard_html,
                "ESG_Dashboard.html",
                "text/html"
            )


if __name__ == "__main__":
    main()
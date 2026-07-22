"""
analytics.py
------------
Plotly visualizations for plagiarism analytics dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Any


def plot_high_severity_trends(trend_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create an interactive line chart showing High severity plagiarism incidents over time.
    
    Args:
        trend_data: List of dicts with 'date' and 'count' keys
        
    Returns:
        Plotly Figure object
    """
    if not trend_data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No High severity incidents recorded in the specified period",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="High Severity Plagiarism Trends (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Number of High Severity Incidents",
            height=400
        )
        return fig
    
    df = pd.DataFrame(trend_data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = px.line(
        df,
        x='date',
        y='count',
        title="High Severity Plagiarism Trends (Last 30 Days)",
        labels={'date': 'Date', 'count': 'Number of High Severity Incidents'},
        markers=True,
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of High Severity Incidents",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    fig.update_traces(
        line=dict(color='#ff4b4b', width=3),
        marker=dict(size=8, color='#ff4b4b')
    )
    
    return fig


def plot_most_plagiarized_documents(doc_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create a bar chart showing the most frequently plagiarized documents.
    
    Args:
        doc_data: List of dicts with 'document_name' and 'incident_count' keys
        
    Returns:
        Plotly Figure object
    """
    if not doc_data:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No plagiarism incidents recorded",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Most Frequently Plagiarized Documents",
            xaxis_title="Document Name",
            yaxis_title="Number of Incidents",
            height=400
        )
        return fig
    
    df = pd.DataFrame(doc_data)
    
    # Truncate long document names for display
    df['display_name'] = df['document_name'].apply(
        lambda x: x[:30] + '...' if len(x) > 30 else x
    )
    
    fig = px.bar(
        df,
        x='display_name',
        y='incident_count',
        title="Most Frequently Plagiarized Documents",
        labels={'display_name': 'Document Name', 'incident_count': 'Number of Incidents'},
        orientation='v',
    )
    
    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Number of Incidents",
        height=400,
        showlegend=False
    )
    
    fig.update_traces(
        marker_color='#ffa500',
        marker_line_color='#cc8400',
        marker_line_width=1.5,
    )
    
    # Add hover template with full document name
    full_names = df['document_name'].tolist()
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Incidents: %{y}<extra></extra>',
        customdata=full_names
    )
    
    # Update hover to show full name
    fig.update_traces(
        hovertemplate='<b>%{customdata}</b><br>Incidents: %{y}<extra></extra>'
    )
    
    return fig

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_gauge_chart(value, title, min_val, max_val):
    """Create a gauge chart for displaying parameter values"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)/3], 'color': "lightgray"},
                {'range': [min_val + (max_val-min_val)/3, min_val + 2*(max_val-min_val)/3], 'color': "gray"},
                {'range': [min_val + 2*(max_val-min_val)/3, max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_feature_importance_plot(importance_scores, feature_names):
    """Create a bar plot for feature importance"""
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    df = df.sort_values('Importance', ascending=True)

    fig = px.bar(df, 
                 x='Importance', 
                 y='Feature',
                 orientation='h',
                 title='Feature Importance in Crop Prediction')

    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Parameters",
        plot_bgcolor='white',
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    fig.update_traces(marker_color='#4CAF50')
    return fig

def create_model_comparison_plot(model_scores):
    """Create a bar plot comparing model performances"""
    models = []
    accuracies = []
    cv_scores = []

    for model_name, scores in model_scores.items():
        models.append(model_name.title())
        accuracies.append(scores['accuracy'])
        cv_scores.append(scores['cv_mean'])

    fig = go.Figure(data=[
        go.Bar(name='Test Accuracy', x=models, y=accuracies, marker_color='#4CAF50'),
        go.Bar(name='Cross-validation Score', x=models, y=cv_scores, marker_color='#2196F3')
    ])

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Accuracy Score',
        plot_bgcolor='white',
        yaxis_tickformat='.0%',
        barmode='group',
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig
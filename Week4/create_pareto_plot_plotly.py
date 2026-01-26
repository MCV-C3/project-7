"""
Create interactive Pareto frontier plot using Plotly and Kaleido
"""
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def load_pareto_data(json_path):
    """Load Pareto frontier data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_pareto_plot(data, output_dir):
    """Create interactive Pareto frontier plot with Plotly"""
    
    # Extract data
    test_accuracies = [trial['test_accuracy'] for trial in data]
    overfitting_values = [trial['overfitting'] for trial in data]
    trial_numbers = [trial['trial_number'] for trial in data]
    
    # Create hover text with hyperparameter details
    hover_texts = []
    for trial in data:
        hyperparams = trial['hyperparameters']
        hover_text = f"<b>Trial {trial['trial_number']}</b><br>"
        hover_text += f"Test Accuracy: {trial['test_accuracy']:.4f}<br>"
        hover_text += f"Overfitting: {trial['overfitting']:.4f}<br><br>"
        hover_text += "<b>Hyperparameters:</b><br>"
        
        for param, value in hyperparams.items():
            if isinstance(value, float):
                if value < 0.001:
                    hover_text += f"{param}: {value:.2e}<br>"
                else:
                    hover_text += f"{param}: {value:.4f}<br>"
            else:
                hover_text += f"{param}: {value}<br>"
        hover_texts.append(hover_text)
    
    # Create the main plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=test_accuracies,
        y=overfitting_values,
        mode='markers+text',
        marker=dict(
            size=12,
            color=trial_numbers,
            colorscale='Viridis',
            colorbar=dict(title="Trial Number"),
            line=dict(width=2, color='white')
        ),
        text=[f"T{num}" for num in trial_numbers],
        textposition="middle center",
        textfont=dict(color='white', size=10),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        name='Trials'
    ))
    
    # Sort points for Pareto front line (by test accuracy)
    sorted_indices = np.argsort(test_accuracies)
    sorted_acc = [test_accuracies[i] for i in sorted_indices]
    sorted_overfitting = [overfitting_values[i] for i in sorted_indices]
    
    # Add Pareto front line
    fig.add_trace(go.Scatter(
        x=sorted_acc,
        y=sorted_overfitting,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Pareto Front',
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Pareto Frontier: Test Accuracy vs Overfitting</b>',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title='<b>Test Accuracy</b>',
            title_font=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='<b>Overfitting (|Train Acc - Test Acc|)</b>',
            title_font=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    # Add annotations for best trials
    best_accuracy = max(test_accuracies)
    best_overfitting = min(overfitting_values)
    
    best_acc_idx = test_accuracies.index(best_accuracy)
    best_overfitting_idx = overfitting_values.index(best_overfitting)
    
    # Annotate best accuracy
    fig.add_annotation(
        x=test_accuracies[best_acc_idx],
        y=overfitting_values[best_acc_idx],
        text=f"Best Accuracy<br>Trial {trial_numbers[best_acc_idx]}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="blue",
        bgcolor="lightblue",
        bordercolor="blue"
    )
    
    # Annotate best overfitting (if different trial)
    if best_acc_idx != best_overfitting_idx:
        fig.add_annotation(
            x=test_accuracies[best_overfitting_idx],
            y=overfitting_values[best_overfitting_idx],
            text=f"Best Overfitting<br>Trial {trial_numbers[best_overfitting_idx]}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            bgcolor="lightgreen",
            bordercolor="green"
        )
    
    # Save interactive HTML
    html_path = os.path.join(output_dir, 'pareto_front_interactive.html')
    fig.write_html(html_path)
    
    # Save static images
    png_path = os.path.join(output_dir, 'pareto_front.png')
    pdf_path = os.path.join(output_dir, 'pareto_front.pdf')
    
    fig.write_image(png_path, width=800, height=600, scale=2)  # High resolution PNG
    fig.write_image(pdf_path, width=800, height=600)  # Vector PDF
    
    # Show the plot
    fig.show()
    
    return fig, html_path, png_path, pdf_path

def print_pareto_summary(data):
    """Print summary of Pareto frontier results"""
    print("=" * 60)
    print("PARETO FRONTIER OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # Sort by test accuracy (descending)
    sorted_data = sorted(data, key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"\nTotal trials on Pareto front: {len(data)}")
    print(f"Best test accuracy: {max(trial['test_accuracy'] for trial in data):.4f}")
    print(f"Best overfitting (lowest): {min(trial['overfitting'] for trial in data):.6f}")
    
    print("\n" + "-" * 60)
    print("DETAILED TRIAL RESULTS:")
    print("-" * 60)
    
    for trial in sorted_data:
        print(f"\n🔹 Trial {trial['trial_number']}")
        print(f"   Test Accuracy: {trial['test_accuracy']:.4f}")
        print(f"   Overfitting:   {trial['overfitting']:.6f}")
        print("   Hyperparameters:")
        
        hyperparams = trial['hyperparameters']
        for param, value in hyperparams.items():
            if isinstance(value, float):
                if value < 0.001:
                    print(f"     {param:<15}: {value:.2e}")
                else:
                    print(f"     {param:<15}: {value:.4f}")
            else:
                print(f"     {param:<15}: {value}")

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'hyperopt', 'pareto_front.json')
    output_dir = os.path.join(base_dir, 'hyperopt', 'plots')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found!")
        return
    
    # Load and process data
    print("📊 Loading Pareto frontier data...")
    data = load_pareto_data(json_path)
    
    # Print summary
    print_pareto_summary(data)
    
    # Create plots
    print(f"\n📈 Creating interactive Pareto frontier plot...")
    fig, html_path, png_path, pdf_path = create_pareto_plot(data, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📄 Interactive HTML: {html_path}")
    print(f"🖼️  High-res PNG:    {png_path}")
    print(f"📑 Vector PDF:      {pdf_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
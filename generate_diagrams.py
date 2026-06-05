"""
Simple script to visualize the technical architecture
Run this after installing dependencies to generate architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create and save the system architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Satellite Error Prediction System Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy'))
    
    # Layer 1: Data Input
    ax.add_patch(FancyBboxPatch((0.5, 9.5), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#90EE90', edgecolor='black', linewidth=2))
    ax.text(2.5, 10.1, 'DATA INPUT LAYER', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 9.75, 'GEO Data', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
    ax.text(3.5, 9.75, 'MEO Data', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
    
    ax.add_patch(FancyBboxPatch((5.5, 9.5), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#90EE90', edgecolor='black', linewidth=2))
    ax.text(7.5, 10.1, 'USER INTERFACE', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 9.75, 'Streamlit Web App', ha='center', fontsize=9)
    
    # Arrow down
    ax.annotate('', xy=(2.5, 9.5), xytext=(2.5, 8.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Layer 2: Preprocessing
    ax.add_patch(FancyBboxPatch((0.5, 7.2), 4, 1.3, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#87CEEB', edgecolor='black', linewidth=2))
    ax.text(2.5, 8.1, 'PREPROCESSING MODULE', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.5, 7.75, '• Outlier Detection & Treatment', ha='center', fontsize=8)
    ax.text(2.5, 7.5, '• Missing Value Handling', ha='center', fontsize=8)
    ax.text(2.5, 7.25, '• Feature Engineering', ha='center', fontsize=8)
    
    # Arrow down
    ax.annotate('', xy=(2.5, 7.2), xytext=(2.5, 6.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Layer 3: Model Training
    ax.add_patch(FancyBboxPatch((0.5, 4.5), 4, 1.7, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFD700', edgecolor='black', linewidth=2))
    ax.text(2.5, 5.9, 'LSTM MODEL ARCHITECTURE', ha='center', fontsize=11, fontweight='bold')
    
    # LSTM layers
    layers = [
        ('LSTM(128) + Dropout(0.3)', 5.6),
        ('LSTM(64) + Dropout(0.3)', 5.35),
        ('LSTM(32) + Dropout(0.2)', 5.1),
        ('Dense(16, ReLU)', 4.85),
        ('Dense(1, Linear)', 4.6)
    ]
    
    for layer_text, y_pos in layers:
        ax.text(2.5, y_pos, layer_text, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow down
    ax.annotate('', xy=(2.5, 4.5), xytext=(2.5, 3.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Layer 4: Training Process
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 4, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFA500', edgecolor='black', linewidth=2))
    ax.text(2.5, 3.2, 'TRAINING PROCESS', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.5, 2.9, 'Adam Optimizer | MSE Loss', ha='center', fontsize=8)
    ax.text(2.5, 2.65, 'Early Stopping | Validation', ha='center', fontsize=8)
    
    # Arrow down
    ax.annotate('', xy=(2.5, 2.5), xytext=(2.5, 1.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Layer 5: Prediction & Evaluation
    ax.add_patch(FancyBboxPatch((0.5, 0.3), 1.8, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FF7F50', edgecolor='black', linewidth=2))
    ax.text(1.4, 1.15, 'PREDICTION', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.4, 0.85, '8th Day Error', ha='center', fontsize=8)
    ax.text(1.4, 0.6, 'Forecasting', ha='center', fontsize=8)
    
    ax.add_patch(FancyBboxPatch((2.7, 0.3), 1.8, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#9370DB', edgecolor='black', linewidth=2))
    ax.text(3.6, 1.15, 'EVALUATION', ha='center', fontsize=10, fontweight='bold')
    ax.text(3.6, 0.85, 'Shapiro-Wilk', ha='center', fontsize=8)
    ax.text(3.6, 0.6, 'RMSE | MAE', ha='center', fontsize=8)
    
    # Right side: Data Flow
    ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 2.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F0F0F0', edgecolor='black', linewidth=2))
    ax.text(7.5, 7.7, 'DATA FLOW', ha='center', fontsize=12, fontweight='bold')
    
    flow_steps = [
        ('1. Load CSV Files', 7.35),
        ('2. Preprocess Data', 7.0),
        ('3. Create Sequences (7→1)', 6.65),
        ('4. Train LSTM Models', 6.3),
        ('5. Make Predictions', 5.95),
        ('6. Evaluate Results', 5.6)
    ]
    
    for step, y_pos in flow_steps:
        ax.text(7.5, y_pos, step, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white'))
    
    # Error Components
    ax.add_patch(FancyBboxPatch((5.5, 3.5), 4, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFE4E1', edgecolor='black', linewidth=2))
    ax.text(7.5, 4.75, 'ERROR COMPONENTS', ha='center', fontsize=11, fontweight='bold')
    
    components = [
        ('X Error (m)', 6.3, 4.4),
        ('Y Error (m)', 8.7, 4.4),
        ('Z Error (m)', 6.3, 3.9),
        ('Clock Error (m)', 8.7, 3.9)
    ]
    
    for comp, x, y in components:
        ax.text(x, y, comp, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Metrics
    ax.add_patch(FancyBboxPatch((5.5, 1.5), 4, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E6E6FA', edgecolor='black', linewidth=2))
    ax.text(7.5, 2.75, 'EVALUATION METRICS', ha='center', fontsize=11, fontweight='bold')
    
    metrics = [
        'RMSE (Root Mean Square Error)',
        'MAE (Mean Absolute Error)',
        'Shapiro-Wilk p-value > 0.05 ✓'
    ]
    
    for idx, metric in enumerate(metrics):
        ax.text(7.5, 2.4 - idx*0.25, metric, ha='center', fontsize=8)
    

    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Architecture diagram saved as 'system_architecture.png'")
    plt.show()


def create_data_flow_diagram():
    """Create and save the data flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'Data Flow & Processing Pipeline', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy'))
    
    # Steps
    steps = [
        (12.5, 'Raw CSV Data\n(GEO/MEO)', '#90EE90'),
        (11.5, 'Parse DateTime\nSort by Time', '#87CEEB'),
        (10.5, 'Handle Missing Values\nLinear Interpolation', '#87CEEB'),
        (9.5, 'Detect Outliers\nIQR Method', '#87CEEB'),
        (8.5, 'Treat Outliers\nCapping', '#87CEEB'),
        (7.5, 'Feature Engineering\nTotal Error, Time Features', '#87CEEB'),
        (6.5, 'Standardization\nStandardScaler', '#FFD700'),
        (5.5, 'Sequence Creation\n7-day windows', '#FFD700'),
        (4.5, 'Train/Val Split\n80% / 20%', '#FFD700'),
        (3.5, 'LSTM Training\n50 epochs', '#FFA500'),
        (2.5, 'Model Prediction\n8th Day', '#FF7F50'),
        (1.5, 'Inverse Transform\nOriginal Scale', '#FF7F50'),
        (0.5, 'Evaluation\nShapiro-Wilk Test', '#9370DB')
    ]
    
    for y, text, color in steps:
        ax.add_patch(FancyBboxPatch((2, y-0.3), 6, 0.6, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor='black', linewidth=2))
        ax.text(5, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if y > 0.5:
            ax.annotate('', xy=(5, y-0.4), xytext=(5, y-0.9),
                        arrowprops=dict(arrowstyle='<-', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Data flow diagram saved as 'data_flow_diagram.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING TECHNICAL DIAGRAMS")
    print("="*60 + "\n")
    
    print("Creating architecture diagram...")
    create_architecture_diagram()
    
    print("\nCreating data flow diagram...")
    create_data_flow_diagram()
    
    print("\n" + "="*60)
    print("✓ ALL DIAGRAMS CREATED SUCCESSFULLY!")
    print("="*60)
    print("\nFiles created:")
    print("  • system_architecture.png")
    print("  • data_flow_diagram.png")
    print("\nYou can use these diagrams in your presentation!")

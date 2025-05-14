import matplotlib.pyplot as plt

# Data
collisions = [0.1, 0.1, 0.1, 10.5, 19.0]
wait_times = [456.32, 200.32, 10.51, 8.9, 4.24]
colors = ['blue', 'orange', 'green', 'red', 'purple']
labels = [
    'Safe weight: 0, Efficiency weight: 1',
    'Safe weight: 0.2, Efficiency weight: 0.7',
    'Safe weight: 0.5, Efficiency weight: 0.5',
    'Safe weight: 0.7, Efficiency weight: 0.2',
    'Safe weight: 1, Efficiency weight: 0'
]

# Create figure
plt.figure(figsize=(10, 6))

# Plot connecting lines (light blue)
plt.plot(collisions, wait_times, color='lightblue', linestyle='-', marker='', zorder=1, linewidth=2)

# Plot scatter points with different colors and add data labels
for x, y, color, label in zip(collisions, wait_times, colors, labels):
    # Plot points
    plt.scatter(x, y, color=color, s=100, label=label, zorder=2, edgecolor='black', linewidth=0.8)
    
    # Add data labels with adjusted positions
    offset = (0.5 if x < 5 else -1.5)  # Adjust x offset based on x position
    plt.annotate(f'({x}, {y})', 
                 (x, y),
                 textcoords="offset points",
                 xytext=(offset, 10),  # Adjust these values to move labels
                 ha='center',
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', lw=0.5, alpha=0.8))

# Add labels and title
plt.xlabel('Collisions', fontsize=12)
plt.ylabel('Wait Times', fontsize=12)
plt.title('Trade-off between Safety and Efficiency', fontsize=14, pad=20)

# Add legend inside plot
plt.legend(loc='upper right', fontsize=9, 
           bbox_to_anchor=(0.98, 0.98),
           framealpha=0.9, edgecolor='gray')


# Adjust layout
plt.tight_layout()

# Save as PDF
plt.savefig('tradeoff.svg', format='svg', bbox_inches='tight', dpi=300)

plt.show()
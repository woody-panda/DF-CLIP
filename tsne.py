import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

# Generate random data for demonstration (replace with your actual t-SNE results)
np.random.seed(42)
n_samples = 100
categories = ['FF++/real', 'FF++/fake', 'Celeb-DF/real', 'Celeb-DF/fake', 
              'DFDC/real', 'DFDC/fake', 'FFIW/real', 'FFIW/fake',
              'DFFD/real', 'DFFD/fake', 'OpenForensics/real', 'OpenForensics/fake',
              'ForgeryNIR/real', 'ForgeryNIR/fake', 'ForgeryNet/real', 'ForgeryNet/fake']
colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('t-SNE Visualization Comparison', y=1.05, fontsize=14)

# Generate and plot synthetic t-SNE data for both methods
for i, cat in enumerate(categories):
    # Pretend these are your t-SNE coordinates - replace with actual data
    x1 = np.random.normal(i%4, 1, n_samples)
    y1 = np.random.normal(i//4, 1, n_samples)
    x2 = np.random.normal(i%4 + 0.5, 0.8, n_samples)
    y2 = np.random.normal(i//4 + 0.5, 0.8, n_samples)
    
    ax1.scatter(x1, y1, color=colors[i], s=10, alpha=0.6, label=cat)
    ax2.scatter(x2, y2, color=colors[i], s=10, alpha=0.6, label=cat)

# Set titles and adjust layout
ax1.set_title('(a) Pre-trained SwinT-B', pad=20)
ax2.set_title('(b) Our method', pad=20)

# Create a single legend above both plots
handles = [mpatches.Patch(color=colors[i], label=cat) for i, cat in enumerate(categories)]
fig.legend(handles=handles, 
           loc='upper center', 
           bbox_to_anchor=(0.5, 1.0),
           ncol=4,
           fontsize=8,
           frameon=False)

plt.tight_layout()
plt.savefig('ef_idd_tsne.pdf', bbox_inches='tight')
plt.show()
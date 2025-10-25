"""
Professional plotting configuration for scientific publications (Nature style).

Features:
- Chinese font support
- Colorblind-friendly palettes
- Publication-quality settings
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from typing import Optional, List, Tuple


# Nature-inspired color palette (colorblind-friendly)
NATURE_COLORS = {
    'primary': ['#0C5DA5', '#00B945', '#FF2C00', '#FF9500', '#845B97', '#474747', '#9e9e9e'],
    'blue': '#0C5DA5',
    'green': '#00B945',
    'red': '#FF2C00',
    'orange': '#FF9500',
    'purple': '#845B97',
    'gray': '#474747'
}


def setup_chinese_font() -> Optional[str]:
    """
    Setup Chinese font for matplotlib.

    Returns:
        Font name if found, None otherwise
    """
    # Common Chinese fonts
    chinese_fonts = [
        'SimHei',  # Windows
        'Microsoft YaHei',  # Windows
        'STHeiti',  # macOS
        'PingFang SC',  # macOS
        'WenQuanYi Micro Hei',  # Linux
        'DejaVu Sans'  # Fallback
    ]

    # Get available fonts
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    # Find first available Chinese font
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return font

    # Try to find any CJK font
    for font_name in available_fonts:
        if any(kw in font_name.lower() for kw in ['cjk', 'hei', 'song', 'chinese']):
            plt.rcParams['font.sans-serif'] = [font_name, 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return font_name

    warnings.warn("No Chinese font found. Chinese text may not display correctly.")
    return None


def setup_nature_style():
    """
    Configure matplotlib for Nature journal style.
    """
    # Setup Chinese font
    chinese_font = setup_chinese_font()

    # Nature-style configuration
    plt.style.use('default')  # Start from default

    plt.rcParams.update({
        # Figure
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',

        # Font sizes
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 8,

        # Axes
        'axes.linewidth': 1.2,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,

        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
    })

    # Set color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', NATURE_COLORS['primary'])

    print("✓ Nature style configured")
    if chinese_font:
        print(f"✓ Chinese font: {chinese_font}")

    return chinese_font


def get_colors(n: int = None) -> List[str]:
    """
    Get N colors from the primary palette.

    Args:
        n: Number of colors (None = all)

    Returns:
        List of hex color codes
    """
    colors = NATURE_COLORS['primary']
    if n is None:
        return colors
    elif n <= len(colors):
        return colors[:n]
    else:
        # Repeat if more needed
        return (colors * ((n // len(colors)) + 1))[:n]


def save_figure(fig, filepath: str, dpi: int = 300, **kwargs):
    """
    Save figure with high quality settings.

    Args:
        fig: Matplotlib figure
        filepath: Output path
        dpi: Resolution
        **kwargs: Additional arguments for savefig
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"✓ Saved: {filepath}")


# Auto-initialize on import
try:
    setup_nature_style()
    matplotlib.interactive(True)  # Enable interactive mode after setup
except Exception as e:
    print(f"Warning: Could not fully initialize plotting config: {e}")


if __name__ == "__main__":
    # Test the configuration
    print("\nTesting Nature-style configuration...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Nature风格可视化测试', fontsize=16, fontweight='bold')

    # Test 1: Line plot
    x = np.linspace(0, 10, 100)
    colors = get_colors(3)
    for i, color in enumerate(colors):
        axes[0, 0].plot(x, np.sin(x + i), label=f'曲线 {i+1}', color=color, linewidth=2)
    axes[0, 0].set_xlabel('X轴')
    axes[0, 0].set_ylabel('Y轴')
    axes[0, 0].set_title('折线图')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Test 2: Bar plot
    categories = ['类别A', '类别B', '类别C', '类别D']
    values = [3, 7, 5, 9]
    axes[0, 1].bar(categories, values, color=get_colors(4), alpha=0.8)
    axes[0, 1].set_xlabel('类别')
    axes[0, 1].set_ylabel('数值')
    axes[0, 1].set_title('柱状图')
    axes[0, 1].grid(True, axis='y')

    # Test 3: Scatter plot
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.5
    axes[1, 0].scatter(x, y, alpha=0.6, color=NATURE_COLORS['blue'], s=50)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('散点图')
    axes[1, 0].grid(True)

    # Test 4: Multiple lines
    for i in range(4):
        y = np.random.randn(20).cumsum()
        axes[1, 1].plot(y, label=f'系列{i+1}', linewidth=2)
    axes[1, 1].set_xlabel('时间')
    axes[1, 1].set_ylabel('数值')
    axes[1, 1].set_title('多系列图')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Save test figure
    from pathlib import Path
    test_dir = Path(__file__).parent.parent.parent / 'results'
    test_dir.mkdir(exist_ok=True)
    save_figure(fig, test_dir / 'plotting_test.png')

    print("\n✓ Configuration test complete!")
    print(f"✓ Test figure saved to: {test_dir / 'plotting_test.png'}")

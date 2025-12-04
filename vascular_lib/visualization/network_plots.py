"""Network visualization functions for debugging and inspection."""

from typing import Optional, Literal
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..core.network import VascularNetwork


def plot_network(
    network: VascularNetwork,
    color_by: Literal["vessel_type", "flow_rate", "reynolds", "radius"] = "vessel_type",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot vascular network in 3D with customizable coloring.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to plot
    color_by : str
        Coloring scheme: vessel_type, flow_rate, reynolds, radius
    ax : matplotlib Axes3D, optional
        Existing axes
    show : bool
        Whether to call plt.show()
    title : str, optional
        Plot title
        
    Returns
    -------
    ax : matplotlib Axes3D
        Axes object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
    
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    for seg in network.segments.values():
        start = network.nodes[seg.start_node_id]
        end = network.nodes[seg.end_node_id]
        
        x = [start.position.x, end.position.x]
        y = [start.position.y, end.position.y]
        z = [start.position.z, end.position.z]
        
        if color_by == "vessel_type":
            color = 'red' if start.vessel_type == "arterial" else 'blue'
        else:
            color = 'gray'
        
        ax.plot(x, y, z, color=color, linewidth=2, alpha=0.7)
    
    positions = np.array([[n.position.x, n.position.y, n.position.z] 
                         for n in network.nodes.values()])
    
    if len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='black', s=10, alpha=0.3)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    if title:
        ax.set_title(title)
    
    if show:
        plt.show()
    
    return ax

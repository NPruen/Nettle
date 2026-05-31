import networkx as nx
import numpy as np
from scipy.optimize import minimize
import json


def loss_function(positions, graph, edge_weight=1.0):
    """
    Loss function for graph embedding.
    Penalizes edges that are not at the target edge_weight distance.
    
    Args:
        positions: Flattened array of vertex positions (shape: n_vertices * d)
        graph: NetworkX graph object
        edge_weight: Target distance between connected vertices
    
    Returns:
        Loss value (float)
    """
    n_vertices = len(graph.nodes())
    d = len(positions) // n_vertices
    positions = positions.reshape(n_vertices, d)
    
    loss = 0.0
    for u, v in graph.edges():
        dist = np.linalg.norm(positions[u] - positions[v])
        loss += (dist - edge_weight) ** 2
    
    return loss


def project_to_2d(positions):
    """
    Project d-dimensional vertex positions to 2D for visualization.
    Uses the first two dimensions.
    
    Args:
        positions: Array of shape (n_vertices, d)
    
    Returns:
        Array of shape (n_vertices, 2)
    """
    return positions[:, :2]


def embed_graph(graph, d=2, num_iterations=1000, learning_rate=0.01):
    """
    Embed a NetworkX graph into d-dimensional space.
    
    Args:
        graph: NetworkX graph object
        d: Dimensionality of embedding space
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
    
    Returns:
        Array of shape (n_vertices, d) with vertex positions
    """
    n_vertices = len(graph.nodes())
    
    # Initialize positions as random unit vectors
    positions = np.random.randn(n_vertices, d)
    positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    
    # Flatten for optimization
    initial_positions = positions.flatten()
    
    # Run optimization
    result = minimize(
        loss_function,
        initial_positions,
        args=(graph, 1.0),
        method='L-BFGS-B',
        options={'maxiter': num_iterations}
    )
    
    # Reshape back to (n_vertices, d)
    optimized_positions = result.x.reshape(n_vertices, d)
    
    return optimized_positions


def create_html_visualization(graph, positions_2d, output_file='graph_visualization.html'):
    """
    Create an HTML visualization of the graph.
    
    Args:
        graph: NetworkX graph object
        positions_2d: Array of shape (n_vertices, 2) with 2D positions
        output_file: Path to output HTML file
    """
    # Normalize positions to fit in a reasonable canvas size
    positions_2d = np.array(positions_2d)
    min_pos = positions_2d.min(axis=0)
    max_pos = positions_2d.max(axis=0)
    range_pos = max_pos - min_pos
    range_pos[range_pos == 0] = 1  # Avoid division by zero
    
    normalized_positions = (positions_2d - min_pos) / range_pos * 800 + 50
    
    # Create HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Graph Visualization</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        canvas {
            border: 1px solid #ccc;
            background-color: white;
        }
    </style>
</head>
<body>
    <canvas id="graphCanvas" width="900" height="900"></canvas>
    <script>
        const canvas = document.getElementById('graphCanvas');
        const ctx = canvas.getContext('2d');
        
        const graph = %GRAPH_DATA%;
        const positions = %POSITIONS_DATA%;
        
        // Draw edges
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        for (let i = 0; i < graph.edges.length; i++) {
            const [u, v] = graph.edges[i];
            const [x1, y1] = positions[u];
            const [x2, y2] = positions[v];
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        
        // Draw vertices
        ctx.fillStyle = '#4CAF50';
        for (let i = 0; i < positions.length; i++) {
            const [x, y] = positions[i];
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw label
            ctx.fillStyle = '#000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i, x, y);
            ctx.fillStyle = '#4CAF50';
        }
    </script>
</body>
</html>"""
    
    # Prepare graph data
    graph_data = {
        'edges': list(graph.edges()),
        'nodes': list(graph.nodes())
    }
    
    # Convert positions to list for JSON
    positions_list = normalized_positions.tolist()
    
    # Replace placeholders
    html_content = html_content.replace('%GRAPH_DATA%', json.dumps(graph_data))
    html_content = html_content.replace('%POSITIONS_DATA%', json.dumps(positions_list))
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.complete_graph(5)
    
    # Embed the graph in 2D space
    print("Embedding graph...")
    positions_d = embed_graph(G, d=4, num_iterations=500)
    
    # Project to 2D (already 2D in this case, but demonstrating the function)
    positions_2d = project_to_2d(positions_d)
    
    # Create HTML visualization
    print("Creating visualization...")
    create_html_visualization(G, positions_2d)
    
    print("Done!")

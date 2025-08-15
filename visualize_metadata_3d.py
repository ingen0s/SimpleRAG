import sys
import chromadb
import networkx as nx
import plotly.graph_objs as go
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Metadata3DViewer(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.setWindowTitle("3D Metadata Map")
        self.resize(1000, 800)
        layout = QVBoxLayout()
        self.label = QLabel("Select a collection:")
        layout.addWidget(self.label)
        self.combo = QComboBox()
        self.collections = self.client.list_collections()
        self.combo.addItems([col.name for col in self.collections])
        layout.addWidget(self.combo)
        self.button = QPushButton("Show 3D Map")
        self.button.clicked.connect(self.show_3d)
        layout.addWidget(self.button)
        self.webview = QWebEngineView()
        layout.addWidget(self.webview)
        self.setLayout(layout)
        if self.collections:
            self.show_3d()

    def show_3d(self):
        idx = self.combo.currentIndex()
        collection = self.collections[idx]
        results = collection.get(include=["metadatas", "documents"])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        if not metadatas or not documents:
            self.webview.setHtml("<h2>No metadata found for this collection.</h2>")
            return
        G = nx.Graph()
        for i, (doc_id, meta) in enumerate(zip(documents, metadatas)):
            node_id = doc_id if doc_id is not None else f"item_{i}"
            G.add_node(node_id, label=str(node_id))
            for k, v in meta.items():
                meta_node = f"{node_id}:{k}"
                G.add_node(meta_node, label=f"{k}: {v}")
                G.add_edge(node_id, meta_node)
        if len(G.nodes) == 0:
            self.webview.setHtml("<h2>No nodes to display in 3D map.</h2>")
            return
        pos = nx.spring_layout(G, dim=3, seed=42)
        x, y, z, text = [], [], [], []
        for node in G.nodes():
            x.append(pos[node][0])
            y.append(pos[node][1])
            z.append(pos[node][2])
            text.append(G.nodes[node]["label"])
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]
        edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=2), hoverinfo='none')
        node_trace = go.Scatter3d(x=x, y=y, z=z, mode='markers+text', marker=dict(size=8, color='blue'), text=text, textposition='top center', hoverinfo='text')
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            title="3D Metadata Map",
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False))
        ))
        html = fig.to_html(include_plotlyjs='cdn')
        self.webview.setHtml(html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    client = chromadb.PersistentClient(path="photo_db")
    window = Metadata3DViewer(client)
    window.show()
    sys.exit(app.exec_())

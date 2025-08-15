import sys
import chromadb
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QLabel, QComboBox

class MetadataTree(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.setWindowTitle("Model Metadata Visualizer")
        self.resize(800, 600)
        layout = QVBoxLayout()
        self.label = QLabel("Select a collection:")
        layout.addWidget(self.label)
        self.combo = QComboBox()
        self.collections = self.client.list_collections()
        self.combo.addItems([col.name for col in self.collections])
        self.combo.currentIndexChanged.connect(self.load_collection)
        layout.addWidget(self.combo)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["ID", "Metadata Key", "Metadata Value"])
        layout.addWidget(self.tree)
        self.setLayout(layout)
        if self.collections:
            self.load_collection(0)

    def load_collection(self, idx):
        self.tree.clear()
        collection = self.collections[idx]
        results = collection.get(include=["metadatas", "documents"])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        for doc_id, meta in zip(documents, metadatas):
            parent = QTreeWidgetItem([str(doc_id), "", ""])
            for k, v in meta.items():
                child = QTreeWidgetItem(["", str(k), str(v)])
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    client = chromadb.PersistentClient(path="photo_db")
    window = MetadataTree(client)
    window.show()
    sys.exit(app.exec_())

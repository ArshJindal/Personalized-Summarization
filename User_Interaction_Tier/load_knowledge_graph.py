import networkx as nx
import pandas as pd

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_interaction(self, user_id, document_id, interaction_type, summary_id=None, generate_summary=False):
        """
        Add interactions between users, documents, and summaries to the graph.

        Parameters:
        - user_id (str): The ID of the user.
        - document_id (str): The ID of the document.
        - interaction_type (str): Type of interaction ('click', 'skip').
        - summary_id (str, optional): The ID of the summary, if applicable.
        - generate_summary (bool): Whether a summary is generated from the document.
        """
        # Adding edges between user and document
        self.graph.add_edge(user_id, document_id, interaction=interaction_type)

        if generate_summary:
            if summary_id is not None:
                # Link document to summary if summary is generated
                self.graph.add_edge(document_id, summary_id, interaction='generate_summary')
                # Link user directly to summary as well
                self.graph.add_edge(user_id, summary_id, interaction='generate_summary')

    def visualize_graph(self):
        """
        Visualizes the graph.
        """
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=7000, node_color='lightblue', font_size=9, font_weight='bold')

def load_data(filepath):
    """
    Load your dataset here. This function is just a placeholder and needs to be adapted
    to how your actual dataset is structured and stored.

    Parameters:
    - filepath (str): Path to the dataset file.

    Returns:
    - pd.DataFrame: The loaded data.
    """
    # Placeholder for actual data loading
    return pd.read_csv(filepath)

def build_graph_from_data(df):
    """
    Builds the graph from the dataset.

    Parameters:
    - df (pd.DataFrame): Data frame containing the user interactions data.
    """
    kg = KnowledgeGraph()
    for index, row in df.iterrows():
        # Example row processing, adapt field names based on your dataset's structure
        user_id = row['user_id']
        document_id = row['document_id']
        interaction_type = row['interaction_type']  # 'click' or 'skip'
        summary_id = row.get('summary_id', None)
        generate_summary = row.get('generate_summary', False)

        kg.add_interaction(user_id, document_id, interaction_type, summary_id, generate_summary)

    return kg

# Example usage
if __name__ == "__main__":
    # Load data
    filepath = 'path_to_your_PENS_dataset.csv'
    data = load_data(filepath)
    
    # Build graph
    knowledge_graph = build_graph_from_data(data)
    
    # Visualize graph
    knowledge_graph.visualize_graph()

import networkx as nx
import matplotlib.pyplot as plt

class TemporalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_interaction(self, user_id, entity_id, interaction_type, time_step):
        """
        Add an interaction to the graph with temporal information.

        Parameters:
        user_id (str): The identifier for the user.
        entity_id (str): The identifier for the entity (document or summary).
        interaction_type (str): Type of interaction ('click', 'skip', 'generate_summary', etc.).
        time_step (int): The time step of the interaction, indicating the sequence order.
        """
        # Add nodes and a directed edge with attributes
        self.graph.add_node(user_id, type='user')
        self.graph.add_node(entity_id, type='document' if 'D' in entity_id else 'summary')
        self.graph.add_edge(user_id, entity_id, interaction=interaction_type, time=time_step)

    def add_summary_generation(self, document_id, summary_id, time_step):
        """
        Specifically add a summary generation interaction.

        Parameters:
        document_id (str): The identifier for the document.
        summary_id (str): The identifier for the summary generated from the document.
        time_step (int): The time step when the summary is generated.
        """
        self.graph.add_node(summary_id, type='summary')
        self.graph.add_edge(document_id, summary_id, interaction='generate_summary', time=time_step)

    def visualize_graph(self):
        """
        Visualize the graph with labels and colors based on type.
        """
        pos = nx.spring_layout(self.graph, seed=42)  # for consistent layout
        color_map = {'user': 'red', 'document': 'blue', 'summary': 'green'}
        colors = [color_map[self.graph.nodes[n]['type']] for n in self.graph.nodes()]

        nx.draw(self.graph, pos, node_color=colors, with_labels=True, arrows=True)
        plt.show()

# Example of using the TemporalKnowledgeGraph class
if __name__ == "__main__":
    tk_graph = TemporalKnowledgeGraph()

    # Simulate adding interactions with time steps
    interactions = [
        ('U1', 'D1', 'click', 1),
        ('U1', 'D2', 'skip', 2),
        ('U1', 'D2', 'generate_summary', 3),
        ('U1', 'S1', 'click', 4),
        ('U2', 'D4', 'click', 5),
        ('U2', 'D6', 'generate_summary', 6),
        ('U2', 'S2', 'click', 7)
    ]

    for user, doc, action, time in interactions:
        tk_graph.add_interaction(user, doc, action, time)
    
    # Example: Generating a summary
    tk_graph.add_summary_generation('D2', 'S1', 3)

    # Visualize the graph
    tk_graph.visualize_graph()

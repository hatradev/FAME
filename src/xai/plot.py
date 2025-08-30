import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re
import pandas as pd
import seaborn as sns

def create_explain_dict(csv_files, top_weight=3, min_keep=3, min_weight=0.0001):
    led = {}
    for feature, file_path in csv_files.items():
        df = pd.read_csv(file_path)
        mean_df = df.mean(numeric_only=True).to_frame(name='weight')
        mean_df.reset_index(inplace=True)
        mean_df.rename(columns={"index": "feature"}, inplace=True)

        if feature == "final_pred":
            sorted_df = mean_df.sort_values(by='weight', ascending=False)
            led[feature] = sorted_df.to_dict(orient='records')
        else:
            filtered_df = mean_df[mean_df['weight'] >= min_weight]
            if len(filtered_df) < min_keep:
                sorted_df = mean_df.sort_values(by='weight', ascending=False)
                led[feature] = sorted_df.head(max(min_keep, top_weight)).to_dict(orient='records')
            else:
                sorted_df = filtered_df.sort_values(by='weight', ascending=False)
                led[feature] = sorted_df.head(top_weight).to_dict(orient='records')
    return led

def plot_leg(led, save_dir):
    G = nx.DiGraph()
    edge_weights = {}
    for target_node, features in led.items():
        target_node_clean = target_node.replace('pred_imf_0_class', 'pred\nimf0\nclass')
        for feature_info in features:
            source_node = feature_info['feature']
            weight = feature_info['weight']
            G.add_edge(source_node, target_node_clean, weight=weight)
            edge_weights[(source_node, target_node_clean)] = weight

    output_node = [node for node, out_degree in G.out_degree() if out_degree == 0]
    input_nodes_unsorted = [node for node, in_degree in G.in_degree() if in_degree == 0]

    def get_lag_number(node_name):
        match = re.search(r'\d+$', node_name)
        return int(match.group()) if match else -1
    input_nodes = sorted(input_nodes_unsorted, key=get_lag_number)

    intermediate_nodes = sorted(list(set(G.nodes()) - set(input_nodes) - set(output_node)))

    pos = {}
    node_gap_x = 1.8 
    for i, node in enumerate(input_nodes):
        pos[node] = (i * node_gap_x, 0)

    start_x_intermediate = (len(input_nodes) - 1) * node_gap_x / 2 - (len(intermediate_nodes) - 1) * 2.5 / 2
    for i, node in enumerate(intermediate_nodes):
        pos[node] = (start_x_intermediate + i * 2.5, 1.2)
 
    if output_node:
        pos[output_node[0]] = ((len(input_nodes) - 1) * node_gap_x / 2, 2.4)

    fig, ax = plt.subplots(figsize=(20, 12))

    color_map = {
        'input': '#6a994e',        
        'intermediate': '#9ccc65', 
        'output': '#386641'        
    }
    node_colors_map = {}
    for node in input_nodes: node_colors_map[node] = color_map['input']
    for node in intermediate_nodes: node_colors_map[node] = color_map['intermediate']
    for node in output_node: node_colors_map[node] = color_map['output']

    positive_color_l1 = '#778da9'
    positive_color_l2 = '#415a77'
    negative_color = '#e63946'
    edge_colors = []
    edge_widths = []

    for u, v in G.edges():
        weight = G.edges[u, v]['weight']
        if u in intermediate_nodes:
            edge_colors.append(negative_color if weight < 0 else positive_color_l2)
        else:
            edge_colors.append(negative_color if weight < 0 else positive_color_l1)
        edge_widths.append(abs(weight) * 25 + 0.8)

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowstyle='-|>', arrowsize=20, node_size=3500
    )

    node_labels = {node: node.replace('_', '\n') for node in G.nodes()}
    for node, (x, y) in pos.items():
        ax.text(x, y, node_labels[node], ha='center', va='center',
                color='white', fontweight='bold', fontsize=15,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=node_colors_map[node], edgecolor='none'))

    edge_labels = {edge: f"{weight:.4f}" for edge, weight in edge_weights.items()}
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=edge_labels,
        font_color='black', font_size=13,
        label_pos=0.5,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
    )

    ax.set_title('Layered Explainability Graph (LEG)', fontsize=24, fontweight='bold')
    ax.margins(x=0.02, y=0.05)
    plt.axis('off')
    positive_influence_upper = mlines.Line2D([], [], color=positive_color_l2, marker='_', markersize=15, label='Positive Influence (upper layer)')
    positive_influence_lower = mlines.Line2D([], [], color=positive_color_l1, marker='_', markersize=15, label='Positive Influence (lower layer)')
    negative_influence = mlines.Line2D([], [], color=negative_color, marker='_', markersize=15, label='Negative Influence')
    ax.legend(handles=[positive_influence_upper, positive_influence_lower, negative_influence], loc='upper right', fontsize=12, frameon=True)

    if save_dir:
        plt.savefig(save_dir)
    return plt

def plot_global_view(csv_files, save_dir):
    df_all = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)

    total_rows = df_all.shape[0]
    feature_freq_pct = (df_all.notna().sum() / total_rows * 100).sort_index()
    feature_mean_pct = (df_all.mean() * 100).sort_index()

    summary_df = pd.DataFrame({
        'Frequency': feature_freq_pct,
        'Mean Importance': feature_mean_pct
    })

    filtered_df = summary_df[
        (summary_df['Frequency'] >= 10) &
        (summary_df['Mean Importance'] >= 1.0)
    ]

    plt.figure(figsize=(6, max(6, int(len(filtered_df) * 0.4))))
    sns.heatmap(filtered_df, cmap="YlGnBu", annot=True, fmt=".1f", cbar=True,
                linewidths=0.5, linecolor='gray', square=False)

    plt.title("Filtered Global View of Feature Importance (in %)")
    plt.xlabel("Metric")
    plt.ylabel("Feature")
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir)
    return plt
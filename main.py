import streamlit as st
import json
import torch
import random

import extra_streamlit_components as stx
from streamlit_extras.stylable_container import stylable_container
from annotated_text import annotated_text

import utils

# ============== CONFIG ==============

st.set_page_config(page_title="GPT2 Neuron Explorer", layout="wide")

model_name = "gpt2-small"
model_path = f"models/{model_name}"

backward_window = 20
forward_window = 10
similar_backward_window = 10
similar_forward_window = 5
importance_context = 21

max_examples = 20

show_activation_values = True

with open(f"{model_path}/config.json") as ifh:
    config = json.load(ifh)

layers = config["layers"]
neurons = config["neurons"]
similarity_thresholds = config["similarity_thresholds"]

# ============== LYRICAL STYLINGS ==============


# st.markdown("""
#             <style>
#                 div[data-testid="column"] {
#                     width: max-content !important;
#                     flex: unset;
#                 }
#                 div[data-testid="column"] * {
#                     width: max-content !important;
#                 }
#             </style>
#             """, unsafe_allow_html=True)


# ============== UTILS ==============


@st.cache_resource
def load():
    with open(f"{model_path}/indexing.json") as ifh:
        indexing = json.load(ifh)
        indexing = {k: utils.to_dict(v) for k, v in indexing.items()}

    with open(f"{model_path}/clusters.json") as ifh:
        clusters = json.load(ifh)
        clusters = utils.to_dict(clusters)

    with open(f"{model_path}/examples.json") as ifh:
        examples = json.load(ifh)

    with open(f"{model_path}/neighbours.json") as ifh:
        neighbours = json.load(ifh)
        neighbours = utils.to_dict(neighbours)

    activations = torch.load(f"{model_path}/activations.pt")

    importances = torch.load(f"{model_path}/importances.pt")

    return activations, indexing, clusters, examples, neighbours, importances


def update_values():
    st.session_state['layer'] = layer
    st.session_state['neuron'] = neuron


# ============== FUNCTIONALITY ==============


def clickable_text(tokens, id_string=""):
    token_cols = st.columns(len(tokens))
    for i, (token, _, colour) in enumerate(tokens):
        with token_cols[i]:
            render_token(token, colour, token_id=f"{id_string}_{i}")


def render_token(token, colour, token_id="0"):
    with stylable_container(
        key=f"token_{token_id}",
        css_styles=f"""
        button {{
            background-color: {colour};
            color: white;
            border: none;
            padding: 0px 6px;
            margin: -4px -8px;
        }}
        """
    ):
        return st.button(token, key=f'button_{token_id}')


def display_neighbours(layer, neuron, feature_idx):
    feature_neighbours = sorted(
        neighbours.get((layer, neuron, feature_idx), []), key=lambda x: x[1], reverse=True
    )

    for (nb_layer, nb_neuron, nb_feature_idx), sim in feature_neighbours:
        if sim < similarity_thresholds[str(layer)]:
            break

        nb_neuron_id = (nb_layer, nb_neuron)
        nb_clusters = clusters.get(nb_neuron_id, [])
        cluster_idx = nb_feature_idx - 1
        nb_feature, (central_idx, _) = nb_clusters[cluster_idx]

        st.write(f"Layer {nb_layer}, Neuron {nb_neuron}, Feature {nb_feature_idx} (similarity: {sim:.2f})")
        display_feature(
            nb_feature, nb_feature_idx, cluster_idxs=[central_idx], backward_window=similar_backward_window, forward_window=similar_forward_window
        )


def display_feature(feature, feature_idx, n_examples=None, cluster_idxs=None, backward_window=backward_window, forward_window=forward_window):
    for i, (cluster_idx, element_idxs) in enumerate(feature):
        if n_examples is not None and i >= n_examples:
            break

        if cluster_idxs is not None and cluster_idx not in cluster_idxs:
            continue

        if len(element_idxs) == 4:
            example_idx, activation_idx, _, importance_idx = element_idxs
        else:
            example_idx, activation_idx, _ = element_idxs
            if show_importance:
                st.write("No importance data available")
                st.write("---")
                return
            importance_idx = 0

        example_tokens = examples[example_idx]
        example_activations = activations[activation_idx]
        example_importances = importances[importance_idx]

        max_activation, max_index = torch.max(example_activations, 0)

        window_start = max(0, max_index - backward_window + 1)
        window_end = max_index + forward_window + 1

        importance_offset = importance_context - backward_window
        to_take_off = max(importance_offset, importance_offset + backward_window - max_index - 1)
        to_add = forward_window

        display_tokens = example_tokens[window_start:window_end]
        display_activations = list(example_activations[window_start:window_end])
        display_importances = list(example_importances[to_take_off:]) + [0] * to_add

        max_importance = example_importances[-1]

        # Normalise by importance of max activating token and clip to 0 and 1
        display_importances = [max(0, min(1, importance / max(0.01, max_importance))) for importance in display_importances]

        if show_importance:
            values = display_importances
            max_value = max_importance
        else:
            values = display_activations
            max_value = max_activation

        display_colours = []
        for value in values:
            pigment = int(255 * max(value, 0) / max_value)
            colour_triple = [str(pigment), "0", "0"] if not show_importance else ["0", "0", str(pigment)]
            display_colours.append(colour_triple)

        coloured_text = [
            (token.replace("$", "\$"), f"{value:.1f}", f"rgb({', '.join(colour)})") if show_activation_values
            else (token, "", f"rgb({', '.join(colour)})")
            for token, value, colour in zip(display_tokens, values, display_colours)
        ]
        annotated_text(coloured_text)

        st.write("---")


def display_neuron(layer, neuron, n_examples=None):
    neuron_id = (layer, neuron)
    neuron_clusters = clusters.get(neuron_id, [])

    if len(neuron_clusters) == 0:
        st.write("No examples available")
        cluster_tabs = []
    else:
        cluster_tabs = st.tabs([f"Feature {i + 1}" for i in range(len(neuron_clusters))])

    for cluster_idx, ((cluster, _), tab) in enumerate(zip(neuron_clusters, cluster_tabs)):
        feature_idx = cluster_idx + 1

        with tab:
            feature_col_width = 0.7
            feature_col, neighbour_col = st.columns([feature_col_width, 1 - feature_col_width])

            with feature_col:
                display_feature(cluster, feature_idx, n_examples=n_examples)

            with neighbour_col:
                st.write(f"### Similar Features")
                display_neighbours(layer, neuron, feature_idx)


# ============== MAIN ==============

activations, indexing, clusters, examples, neighbours, importances = load()

# Initialize or update session state for layer and neuron
if 'layer' not in st.session_state:
    st.session_state['layer'] = 0
if 'neuron' not in st.session_state:
    st.session_state['neuron'] = 0

col1, col2 = st.columns(2)

with col1:
    layer = st.number_input(
        f"Select Layer (0 to {layers - 1})", min_value=0, max_value=layers - 1, value=st.session_state['layer'],
        key='layer_input'
    )
with col2:
    neuron = st.number_input(
        f"Select Neuron (0 to {neurons - 1})", min_value=0, max_value=neurons - 1, value=st.session_state['neuron'],
        key='neuron_input'
    )

width_1, width_2 = 0.07, 0.15
col1, col2, col3 = st.columns([width_1, width_2, 1 - width_1 - width_2])


with col1:
    find = st.button("Find", key="find", on_click=update_values())
with col2:
    lucky = st.button("I'm feeling lucky")
with col3:
    show_importance = st.toggle("Show Token Importance")

if lucky:
    # Update session_state values instead of local variables
    layer = random.randint(0, layers)
    neuron = random.randint(0, neurons)
    update_values()
    st.rerun()

display_neuron(st.session_state['layer'], st.session_state['neuron'], n_examples=max_examples)



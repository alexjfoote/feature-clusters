import streamlit as st
import json
import torch
import random
import uuid
import sqlite3

import numpy as np

from annotated_text import annotated_text
from streamlit_extras.stylable_container import stylable_container

# ============== CONFIG ==============

st.set_page_config(page_title="Feature Clusters", layout="wide")

model_name = "gpt2-small"
model_path = f"models/{model_name}"

backward_window = 20
forward_window = 10
similar_backward_window = 10
similar_forward_window = 5
importance_context = 21
applied_backward_window = 20

max_examples = 20

show_activation_values = True

with open(f"{model_path}/config.json") as ifh:
    config = json.load(ifh)

layers = config["layers"]
neurons = config["neurons"]
similarity_thresholds = config["similarity_thresholds"]

with open("README.md") as ifh:
    readme = "\n".join(ifh.readlines()[4:])


# ============== UTILS ==============

@st.cache_resource
def load():
    conn = sqlite3.connect(f"{model_path}/data.db", check_same_thread=False)
    c = conn.cursor()
    return c


def bytes_to_np(blob):
    return np.frombuffer(blob, dtype=np.float16)


def get(table, _id, converter=json.loads):
    cursor.execute(f"SELECT details FROM {table} WHERE id=?", (_id,))
    value = cursor.fetchone()
    if value is None:
        return []
    value = value[0]
    return converter(value) if converter is not None else value


def to_id(*args):
    return "_".join(str(arg) for arg in args)


def update_previous():
    element = (st.session_state['layer'], st.session_state['neuron'])
    if element != st.session_state.get("current"):
        st.session_state["previous"] = st.session_state.get("current")
        st.session_state["current"] = (st.session_state["layer"], st.session_state["neuron"])


def update_values(new_layer=None, new_neuron=None):
    update_previous()
    st.session_state['layer'] = layer if new_layer is None else new_layer
    st.session_state['neuron'] = neuron if new_neuron is None else new_neuron


def go_back():
    update_values(*st.session_state['previous'])


def clickable_text(text, callback, *args):
    with stylable_container(
        key="clickable",
        css_styles=f"""
        button {{
            background-color: #0e1117;
            color: white;
            border: none;
            cursor: pointer;
            padding: 0!important;
            text-decoration: underline;
        }}
        """
    ):
        st.button(text, on_click=callback, args=args, key=str(uuid.uuid4()))


def escape(token):
    token = token.replace("\\", r"\\")
    token = token.replace("$", "\$")
    return token


# ============== FUNCTIONALITY ==============

def display_neighbours(layer, neuron, feature_idx):
    feature_neighbours = sorted(
        get("neighbours", to_id(layer, neuron, feature_idx)), key=lambda x: x[1], reverse=True
    )

    for (nb_layer, nb_neuron, nb_feature_idx), sim in feature_neighbours:
        if sim < similarity_thresholds[str(layer)]:
            break

        nb_clusters = get("clusters", to_id(nb_layer, nb_neuron))
        cluster_idx = nb_feature_idx - 1
        nb_feature, (central_idx, _) = nb_clusters[cluster_idx]

        neighbour_label = f"Layer {nb_layer}, Neuron {nb_neuron}, Feature {nb_feature_idx}"
        clickable_text(neighbour_label, update_values, nb_layer, nb_neuron)
        display_feature(
            nb_feature, cluster_idxs=[central_idx], backward_window=similar_backward_window, forward_window=similar_forward_window
        )


def display_feature(feature, n_examples=None, cluster_idxs=None, backward_window=backward_window, forward_window=forward_window):
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

        example_tokens = get("examples", example_idx)
        example_activations = torch.tensor(get("activations", activation_idx, bytes_to_np))
        example_importances = torch.tensor(get("importances", importance_idx, bytes_to_np))

        if applied_backward_window is None:
            max_activation, max_index = torch.max(example_activations, 0)
        else:
            offset = example_activations[-1].item()
            example_activations = example_activations[:-1]
            max_activation, max_index = torch.max(example_activations, 0)
            max_index = max_index.item()
            max_index += offset
            max_index = int(max_index)
            window_offset = max(min(applied_backward_window, max_index + 1) - backward_window, 0)

        window_start = max(0, max_index - backward_window + 1)
        window_end = max_index + forward_window + 1

        display_tokens = example_tokens[window_start:window_end]

        if applied_backward_window is None:
            display_activations = list(example_activations[window_start:window_end])
        else:
            display_activations = list(example_activations[window_offset:])

        importance_offset = importance_context - backward_window
        to_take_off = max(importance_offset, importance_offset + backward_window - max_index - 1)
        to_add = forward_window

        display_importances = list(example_importances[to_take_off:]) + [0] * to_add

        max_importance = torch.max(example_importances)

        # Normalise by importance of max activating token and clip to 0 and 1
        display_importances = [max(0, min(1, importance / max(0.01, max_importance))) for importance in display_importances]

        if show_importance:
            values = display_importances
            max_value = 1
        else:
            values = display_activations
            max_value = max_activation

        display_colours = []
        for value in values:
            pigment = int(255 * max(value, 0) / max_value)
            colour_triple = [str(pigment), "0", "0"] if not show_importance else ["0", "0", str(pigment)]
            display_colours.append(colour_triple)

        coloured_text = [
            (escape(token), f"{value:.1f}", f"rgb({', '.join(colour)})") if show_activation_values
            else (token, "", f"rgb({', '.join(colour)})")
            for token, value, colour in zip(display_tokens, values, display_colours)
        ]
        annotated_text(coloured_text)

        st.write("---")


def display_neuron(layer, neuron, n_examples=None):
    neuron_clusters = get("clusters", to_id(layer, neuron))

    if len(neuron_clusters) == 0:
        st.write("No examples available")
        cluster_tabs = []
    else:
        cluster_tabs = st.tabs([f"Feature {i + 1}" for i in range(len(neuron_clusters))])

    for cluster_idx, ((cluster, _), tab) in enumerate(zip(neuron_clusters, cluster_tabs)):
        feature_idx = cluster_idx + 1

        with tab:
            feature_col_width = 0.65
            feature_col, neighbour_col = st.columns([feature_col_width, 1 - feature_col_width])

            with feature_col:
                display_feature(cluster, n_examples=n_examples)

            with neighbour_col:
                st.write(f"### Similar Features")
                st.caption("Click a link to visit that neuron")
                display_neighbours(layer, neuron, feature_idx)


# ============== MAIN ==============

st.title("Feature Clusters")
st.caption("Explore features learned by GPT2-small")
st.caption("Made by Alex Foote")

cursor = load()

explore_tab, about_tab = st.tabs(["Explore", "About"])

with explore_tab:
    if 'layer' not in st.session_state:
        st.session_state['layer'] = 0
    if 'neuron' not in st.session_state:
        st.session_state['neuron'] = 0
    if 'previous' not in st.session_state:
        st.session_state['previous'] = (st.session_state['layer'], st.session_state['neuron'])
    if 'current' not in st.session_state:
        st.session_state['current'] = (st.session_state['layer'], st.session_state['neuron'])

    width = 0.31
    col1, col2, _ = st.columns([width, width, 1 - width - width])

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

    width_1 = 0.13
    col1, col2 = st.columns([width_1, 1 - width_1])

    with col1:
        lucky = st.button("I'm feeling lucky", help="Visit a random neuron")
    with col2:
        backtrack = st.button("Backtrack", help="Go back to the last visited neuron")

    show_importance = st.toggle("Show Token Importance", help="View the importance of each token for neuron activation on the max activating token")

    if lucky:
        layer = random.randint(0, layers - 1)
        neuron = random.randint(0, neurons - 1)
        update_values()
        st.rerun()

    if backtrack:
        go_back()
        st.rerun()

    update_values()
    display_neuron(st.session_state['layer'], st.session_state['neuron'], n_examples=max_examples)

with about_tab:
    st.write(readme)

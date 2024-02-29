# Feature Clusters

https://feature-clusters.streamlit.app/

## Motivation

Understanding MLP neurons in language models is a major challenge in interpretability research. One common approach is to collect the examples in a dataset that most activate a neuron and look for patterns that indicate the neuron’s behaviour. However, many neurons are polysemantic, responding to multiple entirely distinct concepts. This becomes increasingly prevalent as you move beyond the maximally activating dataset examples which are commonly focussed on, with many other behaviours hidden beneath them. 

## Feature Clusters

I tackle some of these problems with **feature clusters**. A feature cluster consists of a set of dataset examples that capture a single behaviour, making it much easier to understand the distinct semantic behaviours of a neuron. 

I collect a wider variety of highly activating examples for each neuron, rather than just the top N, and then cluster these dataset examples into individual features using **feature embeddings**.

## Proof of Concept
You can explore the feature clusters for GPT2-small [here](https://feature-clusters.streamlit.app/). 
Each neuron has a set of features, and each feature has a set of dataset examples.
The tokens in a dataset example are coloured according to their activation, and the activation value is shown next to each token. Each example is focused around a max activating token - the token in the example which caused the highest activation of the neuron.

You can also view token importances for each example. Token importance is the amount of influence the token on has on the activation of the neuron on the max activating token. The more a token affects the activation, the more important it is. This helps to show what parts of the preceding context are required for neuron activation, which is not clear from looking at the activations alone.

Each feature also has a list of similar features from other neurons, ordered by similarity. 

## Technical Details

### Dataset Examples

I ran 10,000 dataset examples from [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) through GPT2-small and collected the internal activations of the model using [TransformerLens](https://github.com/neelnanda-io/TransformerLens).
For each neuron, I collected any example that activated the model more than 0.75x the maximum activation of the neuron (as measured on a much larger dataset, using information from [Neuroscope](https://neuroscope.io/index.html)).

### Feature Embeddings
For each neuron-example pair, I collected the embedding of the max activating token directly before the MLP layer containing the neuron. I then create a **feature embedding** for the example, by taking the Hadamard (element-wise) product of the max token embedding and neuron's input weights.

The intuition behind this is that the pre-neuron embedding of the max activating token is exactly what caused the neuron to activate, and the neuron's input weights are what the neuron is "looking for" in that embedding. The operation to produce the neuron activation is the dot product of the embedding and weights - the sum of the Hadamard product. So the Hadamard product of the embedding and weights is therefore very closely linked to the neuron's activation, and so should be a good representation of the feature that the neuron is responding to.

### Clustering

Given the dataset examples for a neuron and their corresponding feature embeddings, I compute the pairwise cosine similarity between the feature embeddings and create edges between pairs with a similarity above a threshold. I then identify the connected components of the similarity graph. Each connected component forms a feature cluster.

### Token Importance

To compute the importance of each token in the preceding context, I mask the token with a special padding token and compute the change in activation of the neuron on the max activating token. I compute this value for the tokens before the max activating token and for itself, and normalise these by the value for the max activating token. I clip the values to be between 0 and 1, which gives the final importance of each token.

### Similar Features

For each feature cluster, I take the central example - the example with the highest average similarity to all other examples in the cluster.
I take the pre-neuron embedding of the max activating token in this example and perform an approximate nearest neighbours search with [PyNNDescent](https://github.com/lmcinnes/pynndescent) to find the top 20 most similar features, keeping the ones with a similarity above a threshold.

## Contact

If you have any questions, feedback, or are interested in collaborating, feel free to reach out to me at [alexjfoote@icloud.com](mailto:alexjfoote@icloud.com)
# Examples

Every script builds one or more `Picture`s. Render them with the CLI or run them directly:

```bash
pytikz render examples/ml/transformer.py -f pdf,png   # writes next to the script
pytikz tex examples/math/unit_circle.py                # print the generated TikZ
python examples/ml/mlp.py                              # runs the script's __main__ block
python examples/build_all.py                           # everything -> examples/output/
```

* [`basics/`](basics): a minimal script, a tour of the core API and the four themes.
* [`ml/`](ml): architectures (MLP, CNN, Transformer, ResNet block, RNN, VAE, GAN), attention,
  convolution, backpropagation, optimisers, loss landscapes, classical ML plots.
* [`math/`](math): calculus, probability, linear algebra, geometry, dynamical systems,
  graph theory, automata, commutative diagrams and 3D surfaces.

The rendered gallery (PNG, PDF and the generated `.tex` for every example) is in
[`output/`](output/README.md).

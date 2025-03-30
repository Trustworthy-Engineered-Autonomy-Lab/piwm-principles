# Four Principles for Physically Interpretable World Models

This repository contains the source code for this paper: https://arxiv.org/abs/2503.02143

## Prerequisites

```python
pip install -r requirements.txt
```


<!-- ## Data generation



```bash
python gene_data.py 
```

Coming soon.
-->

For principle 1, the separate encoding uses both the standard loss of VAE and extra physical meaningful supervision to compose the latent vectors.

For principle 2, additional translation loss is added to meet the requirement of Equivariant Representation.

For principle 3, velocity estimation is added for the representations under the circumstance of partial supervision.

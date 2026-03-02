This experiment is about knowledge distillation from L-GATr to MLP. 

observation: the scratch mlp in the original kd paper is actually better than our distilled mlp. (them - ~0.9 acc us - ~0.5, ~0.7 after calibrating accuracy)

Hypothesis: 

In our KD script, we fed the MLP the raw, flattened Cartesian 4-momenta (E,px​,py​,pz​). A standard MLP struggles massively to learn Lorentz-invariant physics (like rotations, boosts, and angles) from raw Cartesian coordinates.

However, look closely at what the preprocess() function in the original study does:

    It calculates the total Jet momentum.

    It calculates the Relative Transverse Momentum (rel_pT).

    It calculates the Relative Pseudorapidity (Δη or deta).

    It calculates the Relative Azimuthal Angle (Δϕ or dphi).

They didn't feed the MLP raw 4-momenta; they manually engineered high-level, perfectly rotationally-invariant physics features!  By transforming the data into this relative cylindrical coordinate space, they did 90% of the heavy lifting for the MLP, which is exactly why it was able to hit ~0.9 accuracy.


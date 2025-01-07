package de.c4vxl.models;

import de.c4vxl.engine.activation.Activation;
import de.c4vxl.engine.nn.Linear;
import de.c4vxl.engine.nn.Sequence;
import de.c4vxl.engine.type.DType;

/**
 * This class is a multilayer perceptron (MLP). Utilizing multiple linear transformation and activations to transform an input into a new output.
 * -----
 * This class is an inheritor of the "Sequence"-Module
 * @see de.c4vxl.engine.nn.Sequence
 */
public class MLP extends Sequence {
    public MLP(int in_proj, int out_proj, int num_hidden, int hidden_size) {
        this(in_proj, out_proj, num_hidden, hidden_size, true);
    }

    public MLP(int in_proj, int out_proj, int num_hidden, int hidden_size, boolean bias) {
        this(in_proj, out_proj, num_hidden, hidden_size, bias, Activation.ReLU(), DType.DEFAULT);
    }

    public MLP(int in_proj, int out_proj, int num_hidden, int hidden_size, boolean bias, Activation activation, DType<?> dtype) {
        // input layer + input activation
        this.append(new Linear(in_proj, hidden_size, bias, dtype), activation);

        // hidden layers + activations
        for (int i = 0; i < num_hidden; i++)
            this.append(new Linear(hidden_size, hidden_size, bias, dtype), activation);

        // output layer
        this.append(new Linear(hidden_size, out_proj, bias, dtype));
    }
}
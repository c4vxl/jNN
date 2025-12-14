package de.c4vxl.models;

import de.c4vxl.core.nn.Linear;
import de.c4vxl.core.nn.activation.type.Activation;
import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.type.DType;

import java.util.ArrayList;

/**
 * This class is a multilayer perceptron (MLP). Utilizing multiple linear transformation and activations to transform an input into a new output.
 */
public class MLP extends Module {
    public ArrayList<Linear> layers = new ArrayList<>();
    public Activation activation;

    public MLP(int in_proj, int out_proj, int num_hidden, int hidden_size) {
        this(in_proj, out_proj, num_hidden, hidden_size, true);
    }

    public MLP(int in_proj, int out_proj, int num_hidden, int hidden_size, boolean bias) {
        this(in_proj, out_proj, num_hidden, hidden_size, bias, Activation.ReLU(), DType.DEFAULT);
    }

    public MLP(int in_proj, int out_proj, int num_hidden, int hidden_size, boolean bias, Activation activation, DType<?> dtype) {
        this.activation = activation;

        // input layer + input activation
        this.layers.add(new Linear(in_proj, hidden_size, bias, dtype));

        // hidden layers + activations
        for (int i = 0; i < num_hidden; i++)
            this.layers.add(new Linear(hidden_size, hidden_size, bias, dtype));

        // output layer
        this.layers.add(new Linear(hidden_size, out_proj, bias, dtype));
    }

    /**
     * Forward through the mlp
     * @param input The inputs to the input layer
     */
    public <T> Tensor<T> forward(Tensor<T> input) {
        boolean wasUnsqueezed = false;
        if (input.dim() == 1) {
            wasUnsqueezed = true;
            input = input.unsqueeze(0);
        }

        // Forward through layers
        Tensor<T> result = input;
        int numLayers = this.layers.size();
        for (int i = 0; i < numLayers; i++) {
            // Forward through linear layer
            result = this.layers.get(i).forward(result);

            // Skip activation for last layer
            if (i == numLayers - 1)
                continue;

            // Forward through activation
            result = this.activation.forward(result);
        }

        if (wasUnsqueezed)
            return result.squeeze(0);

        return result;
    }
}
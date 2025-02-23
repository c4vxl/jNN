package de.c4vxl.models;

import de.c4vxl.engine.nn.Embedding;
import de.c4vxl.engine.nn.Linear;
import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.type.DType;
import de.c4vxl.models.type.TextGenerationModel;

/**
 * This is a sequence utilizing a long-short term memory (LSTM) model for natural language processing
 * This class sequences an Embedding, multiple LSTM and an output projection behind each other
 */
public class LSTMForNLP extends TextGenerationModel {
    public LSTMForNLP(int vocab_size, int n_embd, int hidden_size, int num_layers) {
        this(vocab_size, n_embd, hidden_size, num_layers, true, DType.DEFAULT);
    }

    public Embedding embedding;
    public LSTM lstm;
    public Linear out_proj;

    public LSTMForNLP(int vocab_size, int n_embd, int hidden_size, int num_layers, boolean bias, DType<?> dtype) {
        this.embedding = new Embedding(vocab_size, n_embd);
        this.lstm = new LSTM(n_embd, hidden_size, num_layers, bias, dtype);
        this.out_proj = new Linear(hidden_size, vocab_size);
    }

    @Override
    public <T extends Number> Tensor<Double> forward(Tensor<T> input) {
        Tensor<Double> x = this.embedding.forward(input).asDouble();
        x = this.lstm.forward(x);
        x = this.out_proj.forward(x);
        return x;
    }
}
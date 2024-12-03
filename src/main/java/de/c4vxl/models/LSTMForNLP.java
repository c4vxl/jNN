package de.c4vxl.models;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.nn.Embedding;
import de.c4vxl.engine.nn.Linear;
import de.c4vxl.models.type.TextGenerationModel;

import java.util.List;

@SuppressWarnings({"rawtypes", "unchecked"})
public class LSTMForNLP extends TextGenerationModel {
    public Embedding embedding;
    public LSTM lstm;
    public Linear fc;
    private LSTM.LSTMOutput last_output = null;

    public LSTMForNLP(int vocab_size, int n_embd, int hidden_size, int num_layers) { this(vocab_size, n_embd, hidden_size, num_layers, 0, true); }
    public LSTMForNLP(int vocab_size, int n_embd, int hidden_size, int num_layers, int proj_size, boolean bias) {
        this.embedding = new Embedding(vocab_size, n_embd);
        this.lstm = new LSTM(n_embd, hidden_size, num_layers, proj_size, bias);
        this.fc = new Linear(hidden_size, vocab_size);
    }

    @Override
    public <T extends Number> Tensor<T> forward(Tensor<T> input) {
        if (last_output != null)
            return forward(input, this.last_output.h, this.last_output.c);

        return forward(input, null, null);
    }
    
    public <T extends Number> Tensor<T> forward(Tensor<T> input, List<Tensor<T>> h, List<Tensor<T>> c) {
        input = this.embedding.forward(input).transpose(1, 0, 2); // embedding
        LSTM.LSTMOutput<T> lstm_out = this.lstm.forward(input, h, c); // lstm
        
        this.last_output = new LSTM.LSTMOutput(
                this.fc.forward(lstm_out.output), // output projection
                lstm_out.h,
                lstm_out.c
        );
        
        return this.last_output.output;
    }
}
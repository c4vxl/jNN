package de.c4vxl.engine.nn;

import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.data.Tensor;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Embedding extends Module {
    private final Tensor embedding;

    public Embedding(int vocab_size, int embedding_dim) {
        this.embedding = Tensor.ones(vocab_size, embedding_dim);
    }

    public <T> Tensor<T> forward(Tensor<T> tokenIndices) {
        int batchSize = tokenIndices.size(0);
        int seqLength = tokenIndices.size(1);
        int n_embd = this.embedding.size(1);

        Tensor output = Tensor.zeros(this.embedding.dtype, batchSize, seqLength, n_embd);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLength; t++) {
                int index = (int) tokenIndices.item(b, t);
                for (int e = 0; e < n_embd; e++) {
                    output.set(embedding.item(index, e), b, t, e);
                }
            }
        }

        return output;
    }
}
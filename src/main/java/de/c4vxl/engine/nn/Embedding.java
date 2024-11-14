package de.c4vxl.engine.nn;

import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.data.Tensor;

@SuppressWarnings({"rawtypes", "unchecked"})
public class Embedding extends Module {
    public final Tensor embedding;

    public Embedding(int vocab_size, int embedding_dim) {
        this.embedding = Tensor.ones(Tensor.defaultDataType, vocab_size, embedding_dim);
    }

    public <T> Tensor<T> forward(Tensor<T> tokenIndices) {
        if (tokenIndices.dtype != Tensor.defaultDataType)
            throw new IllegalArgumentException("Embedding only works with Tensor.defaultDataType as of right now!");

        int batchSize = tokenIndices.size(0);
        int seqLength = tokenIndices.size(1);
        int n_embd = this.embedding.size(1);

        Tensor output = Tensor.zeros(this.embedding.dtype, batchSize, seqLength, n_embd);

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                int tokenIndex = tokenIndices.item(Integer.class, i, j);

                output.set(embedding.item(tokenIndex), i, j);
            }
        }

        tokenIndices.data = (T[]) output.data;

        return output;
    }
}
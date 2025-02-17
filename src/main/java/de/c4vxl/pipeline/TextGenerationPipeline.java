package de.c4vxl.pipeline;

import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.models.type.TextGenerationModel;
import de.c4vxl.pipeline.type.Pipeline;
import de.c4vxl.tokenizers.type.Tokenizer;

public class TextGenerationPipeline extends Pipeline {
    public Tokenizer tokenizer;
    public TextGenerationModel model;
    public int blockSize = 1024;
    public double temperature = 1.0;
    public int newTokens = 100;

    public TextGenerationPipeline(Tokenizer tokenizer, TextGenerationModel model) {
        this.tokenizer = tokenizer;
        this.model = model;
    }

    /**
     * Change the default block size
     */
    public TextGenerationPipeline blockSize(Integer newVal) {
        this.blockSize = newVal;
        return this;
    }

    /**
     * Change the default temperature
     */
    public TextGenerationPipeline temperature(double newVal) {
        this.temperature = newVal;
        return this;
    }

    /**
     * Change the default max_new_tokens
     */
    public TextGenerationPipeline newTokens(Integer newVal) {
        this.newTokens = newVal;
        return this;
    }

    public String forward(String prompt) { return this.forward(prompt, this.newTokens, null); }
    public String forward(String prompt, int max_new_tokens) { return this.forward(prompt, max_new_tokens, null); }
    public String forward(String prompt, int max_new_tokens, TextGenerationModel.GenerationStream stream) { return this.forward(prompt, this.temperature, max_new_tokens, this.blockSize, stream); }
    public String forward(String prompt, double temperature, int max_new_tokens, int block_size, TextGenerationModel.GenerationStream stream) {
        Tensor<Integer> tokenized = this.tokenizer.forward(prompt);
        Tensor<Integer> output = model.generate(tokenized, temperature, max_new_tokens, block_size, this.tokenizer.eosTokenID(), stream);
        return this.tokenizer.decode(output);
    }
}

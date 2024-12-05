package de.c4vxl.pipeline;

import de.c4vxl.engine.module.Module;
import de.c4vxl.models.type.TextGenerationModel;
import de.c4vxl.tokenizers.Tokenizer;

public class LanguageModelPipeline extends Module {
    public Tokenizer tokenizer;
    public TextGenerationModel model;

    public LanguageModelPipeline(TextGenerationModel mdl, Tokenizer tkn) {
        this.model = mdl;
        this.tokenizer = tkn;
    }

    public String forward(String input) { return forward(input, 50, null, 50, null); }
    public String forward(String input, int max_new_tokens, TextGenerationModel.GenerationStream stream, int block_size, Integer eos_token_id) {
        return tokenizer.decode(model.generate(tokenizer.encode(input), max_new_tokens, stream, block_size, eos_token_id));
    }
}

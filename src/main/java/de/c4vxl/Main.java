package de.c4vxl;

import de.c4vxl.models.LSTMForNLP;
import de.c4vxl.pipeline.LanguageModelPipeline;
import de.c4vxl.tokenizers.BPETokenizer;

public class Main {
    public static void main(String[] args) {
        BPETokenizer tokenizer = BPETokenizer.load("tokenizer");
        assert tokenizer != null;

        LanguageModelPipeline pipeline = new LanguageModelPipeline(
                new LSTMForNLP(tokenizer.vocab_size(), 64, 64, 2),
                tokenizer
        );

        System.out.println(pipeline.forward("Hello"));
    }
}
package de.c4vxl.tokenizers.type;

import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.tensor.Tensor;

import java.util.Map;

public abstract class Tokenizer extends Module {
    public String unk_token, eos_token, pad_token, bos_token;

    public Tokenizer(String unk_token, String eos_token, String pad_token, String bos_token) {
        this.unk_token = unk_token;
        this.eos_token = eos_token;
        this.pad_token = pad_token;
        this.bos_token = bos_token;
    }

    public Integer unkTokenID() { return convertTokenToId(eos_token); }
    public Integer eosTokenID() { return convertTokenToId(eos_token); }
    public Integer padTokenID() { return convertTokenToId(eos_token); }
    public Integer bosTokenID() { return convertTokenToId(eos_token); }

    public abstract Integer[] encode_(String text);
    public abstract String decode_(Integer[] tokens);
    public abstract int vocabSize();
    public abstract int convertTokenToId(String token);
    public abstract String convertIdToToken(Integer token);
    public abstract Map<String, Integer> getVocab();

    public Tensor<Float> encode(String text) { return Tensor.of(encode_(text)).asFloat().unsqueeze(0); }
    public String decode(Tensor<?> text) { return decode_(text.squeeze().asInt().data); }

    public Tensor<Float> forward(String text) {
        return this.encode(text);
    }
}
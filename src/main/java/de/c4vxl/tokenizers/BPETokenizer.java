package de.c4vxl.tokenizers;

import de.c4vxl.engine.module.Module;

import java.nio.charset.StandardCharsets;
import java.util.*;

public class BPETokenizer extends Module implements Tokenizer {
    public static class BytePair {
        public Integer a;
        public Integer b;
        public BytePair(Integer a, Integer b) {
            this.a = a;
            this.b = b;
        }
    }

    public HashMap<BytePair, Integer> vocab;


    public BPETokenizer(HashMap<BytePair, Integer> vocab) {
        this.vocab = vocab;
    }

    public int vocab_size() { return vocab.size(); }


    // convert text to utf8-byte-Integers
    public static Integer[] textToByteID(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        Integer[] ids = new Integer[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            ids[i] = bytes[i] & 0xFF; // ensure unsigned conversion of bytes to integers
        }

        return ids;
    }

    // assign a specific byte-pair a new id
    public static Integer[] mergePair(Integer[] ids, BytePair pair, int newID) {
        ArrayList<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.length) {
            if (i < ids.length - 1 && ids[i].equals(pair.a) && ids[i + 1].equals(pair.b)) {
                newids.add(newID);
                i += 2;
            } else {
                newids.add(ids[i]);
                i += 1;
            }
        }

        return newids.toArray(new Integer[0]);
    }

    // count the occurrences of each byte-pair
    public static HashMap<BytePair, Integer> getPairCount(Integer[] ids) {
        // I am using a LinkedMap here, because apparently java automatically sorts normal HashMaps by its keys
        // this could turn a Map like { BytePair[a=72, b=101] : 3, BytePair[a=104, b=111] : 3 } to { BytePair[a=104, b=111] : 3, BytePair[a=72, b=101] : 3 } by itself
        // this could cause some unexpected behaviours
        HashMap<BytePair, Integer> counts = new LinkedHashMap<>();
        for (int i = 0; i < ids.length - 1; i++) {
            BytePair pair = new BytePair(ids[i], ids[i + 1]);
            counts.put(pair, counts.getOrDefault(pair, 0) + 1);
        }
        return counts;
    }

    public HashMap<Integer, String> generateTokenToBytesMap() {
        HashMap<Integer, String> newVocab = new HashMap<>() {{
            for (Integer idx = 0; idx < 255; idx++)
                put(idx, new String(new byte[]{idx.byteValue()}));
        }};

        int idx = 0;
        for (Map.Entry<BytePair, Integer> entry : this.vocab.entrySet().stream().sorted(Map.Entry.comparingByValue()).toList()) {
            newVocab.put(idx, newVocab.get(entry.getKey().a) +
                    newVocab.get(entry.getKey().b));
            idx++;
        }

        return newVocab;
    }

    // create a BPETokenizer instance trained on a specific training text
    public static BPETokenizer train(String text, int target_vocab_size) {
        Integer[] ids = textToByteID(text);

        HashMap<BytePair, Integer> vocab = new HashMap<>();
        for (int i = 0; i < target_vocab_size; i++) {
            HashMap<BytePair, Integer> stats = getPairCount(ids);
            if (stats.isEmpty()) break;

            BytePair pair = Collections.max(
                    stats.entrySet(),
                    Comparator.comparingInt(Map.Entry::getValue)
            ).getKey();

            ids = mergePair(ids, pair, i);
            vocab.put(pair, i);
        }

        return new BPETokenizer(vocab);
    }

    @Override
    public List<Integer> _encode(String input) {
        Integer[] ids = textToByteID(input);

        while (ids.length >= 2) {
            HashMap<BytePair, Integer> stats = getPairCount(ids);
            BytePair pair = Collections.min(
                    stats.entrySet(),
                    Comparator.comparingInt(entry -> vocab.getOrDefault(entry.getKey(), Integer.MAX_VALUE))
            ).getKey();

            if (!this.vocab.containsKey(pair))
                break;

            ids = mergePair(ids, pair, this.vocab.get(pair));
        }

        return Arrays.stream(ids).toList();
    }

    @Override
    public String _decode(List<Integer> input_ids) {
        HashMap<Integer, String> reverse_vocab = generateTokenToBytesMap();
        StringBuilder output = new StringBuilder();

        for (Integer idx : input_ids)
            output.append(
                    Objects.requireNonNullElse(reverse_vocab.get(idx),
                            "<|UNKNOWN|>")
            );

        return output.toString();
    }
}
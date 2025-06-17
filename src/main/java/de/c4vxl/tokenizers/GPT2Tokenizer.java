package de.c4vxl.tokenizers;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import de.c4vxl.core.module.Module;
import de.c4vxl.tokenizers.type.Tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This is an implementation of the GPT2 Tokenizer as seen
 * @see <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py">Here</a>
 */
@SuppressWarnings("ClassEscapesDefinedScope")
public class GPT2Tokenizer extends Tokenizer {
    public record Pair<A, B>(A first, B second) {
        @Override
        public boolean equals(Object o) {
            return (o instanceof Pair) && Objects.equals(((Pair<?, ?>) o).first, this.first) && Objects.equals(((Pair<?, ?>) o).second, this.second);
        }

        @Override
            public String toString() {
                return "Pair{" +
                        "first=" + first +
                        ", second=" + second +
                        '}';
            }
        }

    /**
     * Generates a mapping of utf-8 byte to unicode strings.
     */
    public static Map<Integer, String> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>(){{
            for (int i = '!'; i < '~' + 1; i++) add(i);
            for (int i = '¡'; i < '¬' + 1; i++) add(i);
            for (int i = '®'; i < 'ÿ' + 1; i++) add(i);
        }};

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!bs.contains(b)) {
                bs.add(b); cs.add(256 + n);
                n++;
            }

        List<Character> csChars = new ArrayList<>(){{ for (int num : cs) add((char) num); }};

        return new LinkedHashMap<>(){{
            for (int i = 0; i < bs.size(); i++)
                put(bs.get(i), String.valueOf(csChars.get(i)));
        }};
    }

    /**
     * Return set of symbol pairs in a word.
     */
    public static Set<Pair<String, String>> getPairs(List<String> word) {
        Set<Pair<String, String>> pairs = new LinkedHashSet<>();
        String prev = word.getFirst();
        for (int i = 1; i < word.size(); i++) {
            String current = word.get(i);
            pairs.add(new Pair<>(prev, current));
            prev = current;
        }

        return pairs;
    }

    private Map<String, Integer> encoder;
    private Map<Integer, String> decoder;
    private Map<Integer, String> byteEncoder;
    private Map<String, Integer> byteDecoder;
    private Map<Pair<String, String>, Integer> bpeRanks;
    private Map<String, String> cache;
    private Pattern pattern;
    private List<Pair<String, String>> merges;

    /**
     * Creates an un-initialized version of the tokenizer.
     */
    public GPT2Tokenizer() {
        super("<|endoftext|>", "<|endoftext|>", "<|endoftext|>", "<|endoftext|>");
    }

    /**
     * Create a tokenizer instance and initialize it.
     * @param encoder The vocabulary of the tokenizer
     * @param bpeMerges The merges history
     */
    public GPT2Tokenizer(Map<String, Integer> encoder, List<Pair<String, String>> bpeMerges) {
        super("<|endoftext|>", "<|endoftext|>", "<|endoftext|>", "<|endoftext|>");
        init(encoder, bpeMerges);
    }

    private void init(Map<String, Integer> encoder, List<Pair<String, String>> bpeMerges) {
        // set vocabulary en-/decoder
        this.encoder = encoder;
        this.decoder = new HashMap<>(){{ encoder.forEach((k, v) -> put(v, k)); }};

        // generate byte-unicode en-/decoder
        this.byteEncoder = bytesToUnicode();
        this.byteDecoder = new HashMap<>(){{ byteEncoder.forEach((k, v) -> put(v, k)); }};

        // Mapping: bpeMerges -> $i
        this.merges = bpeMerges;
        this.bpeRanks = new HashMap<>(){{ for (int i = 0; i < bpeMerges.size(); i++) put(bpeMerges.get(i), i); }};

        this.cache = new HashMap<>();
        this.pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");;
    }

    /**
     * Load a tokenizer from its data dir
     * @param dataDir The path to the dir containing the vocab.json and merges.txt
     */
    public static GPT2Tokenizer from_pretrained(String dataDir) {
        return GPT2Tokenizer.from_pretrained(Path.of(dataDir, "vocab.json"), Path.of(dataDir, "merges.txt"));
    }

    /**
     * Loads a tokenizer from a vocab and merges file
     * @param vocab_file The path to the vocab file
     * @param merges_file The path to the merges file
     */
    public static GPT2Tokenizer from_pretrained(Path vocab_file, Path merges_file) {
        try {
            return new GPT2Tokenizer(
                    new Gson().fromJson(Files.readString(vocab_file), new TypeToken<Map<String, Integer>>(){}.getType()),
                    new ArrayList<>(
                            Files.readAllLines(merges_file).stream()
                                    .map(l -> l.split("\\s+"))
                                    .filter(parts -> parts.length >= 2 && !parts[0].startsWith("#"))
                                    .map(parts -> new Pair<>(parts[0], parts[1])).toList()
                    )
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Compute the BPE for a token
     * @param token The token to encode
     */
    public String bpe(String token) {
        // if cached, just use it
        if (this.cache.containsKey(token))
            return this.cache.get(token);

        // split into characters
        List<String> word = new ArrayList<>(List.of(token.split("")));

        Set<Pair<String, String>> pairs = getPairs(word);
        if (pairs.isEmpty())
            return token;

        while (true) {
            Pair<String, String> bigram = pairs.stream()
                    .min(Comparator.comparingInt(p -> this.bpeRanks.getOrDefault(p, Integer.MAX_VALUE)))
                    .orElseGet(null);

            if (!this.bpeRanks.containsKey(bigram))
                break;

            String first = bigram.first, second = bigram.second;
            List<String> newWord = new ArrayList<>();
            int i = 0;
            while (i < word.size()) {
                int j = word.subList(i, word.size()).indexOf(first); // 'word.index(first, i)'

                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                } else {
                    newWord.addAll(word.subList(i, j));
                    i = j;
                }

                if (word.get(i).equals(first) && i < word.size() - 1 && word.get(i + 1).equals(second)) {
                    newWord.add(first + second);
                    i += 2;
                } else {
                    newWord.add(word.get(i));
                    i += 1;
                }
            }

            word = newWord;

            if (word.size() == 1) break;
            else pairs = getPairs(word);
        }

        String out = String.join(" ", word);
        cache.put(token, out);

        return out;
    }

    /**
     * Split a text into their "tokens"
     * @param text The text to tokenize
     */
    public List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        Matcher matcher = this.pattern.matcher(text);
        while (matcher.find()) {
            // convert to bytes
            String token = matcher.group();

            // encode using byteEncoder
            StringBuilder encoded = new StringBuilder();

            for (byte b : token.getBytes(StandardCharsets.UTF_8)) {
                encoded.append(this.byteEncoder.get((int) b));
            }

            // Apply byte pair encoding
            String[] bpeTokens = bpe(encoded.toString()).split(" ");
            tokens.addAll(Arrays.asList(bpeTokens));
        }

        return tokens;
    }

    /**
     * Convert a string token to its id
     * If no representation exists, the representation of this.unk_token will be returned
     * @param token The string token
     */
    @Override public int convertTokenToId(String token) {
        return this.encoder.getOrDefault(token, this.encoder.get(this.unk_token));
    }

    /**
     * Convert an int token to its string
     * @param token The int token
     */
    @Override public String convertIdToToken(Integer token) {
        return this.decoder.get(token);
    }

    /**
     * Converts a sequence of string tokens in a single string
     * @param tokens The sequence of tokens
     */
    public String convertTokensToString(String[] tokens) {
        String text = String.join("", tokens);

        byte[] decodedBytes = new byte[text.length()];
        for (int i = 0; i < text.length(); i++)
            decodedBytes[i] = this.byteDecoder.get(String.valueOf(text.charAt(i))).byteValue();

        return new String(decodedBytes, StandardCharsets.UTF_8);
    }

    @Override public int vocabSize() { return this.encoder.size(); }

    @Override public Map<String, Integer> getVocab() { return new HashMap<>(this.encoder); }

    @Override
    public Integer[] encode_(String text) {
        return this.tokenize(text).stream().map(this::convertTokenToId).toArray(Integer[]::new);
    }

    @Override
    public String decode_(Integer[] tokens) {
        return this.convertTokensToString(Arrays.stream(tokens).map(this::convertIdToToken).toArray(String[]::new));
    }

    @Override
    public Map<String, Object> state() {
        return new HashMap<>() {{
            put("encoder", encoder);
            put("merges", new ArrayList<>(merges.stream().map((pair) -> pair.first + ";" + pair.second).toList()));
        }};
    }

    @SuppressWarnings("unchecked")
    @Override
    public <T extends Module> T load_state(Map<String, Object> state) {
        // Ensure all important keys are present
        if (!(state.containsKey("merges") && state.containsKey("encoder"))) return (T) this;

        // Get encoder
        Map<String, Object> _encoder = (Map<String, Object>) state.get("encoder");
        Map<String, Integer> encoder = new HashMap<>() {{
            _encoder.forEach((key, value) -> {
                if (value instanceof Double d) value = d.intValue();
                else value = Integer.valueOf(value.toString());
                put(key, (Integer) value);
            });
        }};

        // Get merges history
        List<Pair<String, String>> bpeMerges = ((ArrayList<String>) state.get("merges")).stream()
                .map(x -> {
                    String[] parts = x.split(";");
                    if (parts.length != 2) return null;
                    return new Pair<>(parts[0], parts[1]);
                }).filter(Objects::nonNull).toList();

        // Init
        init(encoder, bpeMerges);

        return (T) this;
    }
}

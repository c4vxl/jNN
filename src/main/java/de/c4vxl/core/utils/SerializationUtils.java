package de.c4vxl.core.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.Strictness;
import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.type.DType;
import de.c4vxl.jNN;
import org.nd4j.shade.guava.reflect.TypeToken;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.util.*;

/**
 * A utility class for the serialization of modules.
 * @see de.c4vxl.core.nn.module.Module
 */
public class SerializationUtils {
    /**
     * Generate a state of an object
     * @param object The object
     * @param state A map to generate the state into
     * @param prefix A prefix for all paths (leave as "")
     */
    public static void generateStateRecursively(Object object, Map<String, Object> state, String prefix) {
        try {
            if (object != null && (object instanceof de.c4vxl.core.nn.module.Module ||
                    object instanceof List<?> ||
                    object instanceof Map<?,?> ||
                    object.getClass().isArray())) {

                for (Field field : object.getClass().getFields()) {
                    if (Modifier.isPrivate(field.getModifiers()) || field.isSynthetic()) continue;

                    field.setAccessible(true);

                    String name = field.getName();
                    Object value = field.get(object);

                    // submodules
                    if (Module.class.isAssignableFrom(field.getType()))
                        generateStateRecursively(value, state, prefix + name + ".");

                    // lists
                    else if (List.class.isAssignableFrom(field.getType())) {
                        for (int i = 0; i < ((List<?>) value).size(); i++)
                            generateStateRecursively(((List<?>) value).get(i), state, prefix + name + "." + i + ".");
                    }

                        // maps
                    else if (Map.class.isAssignableFrom(field.getType()))
                        ((Map<?, ?>) value).forEach((key, val) ->
                                generateStateRecursively(val, state, prefix + name + "." + key + "."));

                        // arrays
                    else if (field.getType().isArray())
                        for (int i = 0; i < Array.getLength(value); i++)
                            generateStateRecursively(Array.get(value, i), state, prefix + name + "." + i + ".");

                        // parameter
                    else
                        generateStateRecursively(value, state, prefix + name);
                }
            } else
                state.put(prefix.endsWith(".") ? prefix.substring(0, prefix.length() - 1) : prefix, object);
        } catch (Exception e) {
            if (jNN.LOG_STATE_GENERATION_ERROR)
                System.err.println("WARNING: Error while generating state. " + e);
        }
    }

    /**
     * Update values in the module based on a specific state
     * @param object The module
     * @param last The last module in the sequence (set to null)
     * @param f The field of the module (set to null)
     * @param state The state to load from
     * @param prefix The prefix of the current object (set to "")
     */
    @SuppressWarnings("unchecked")
    public static <T> void loadStateRecursively(T object, Object last, Field f, Map<String, Object> state, String prefix) {
        if (object == null)
            return;

        String trimmedPrefix = prefix.endsWith(".") ? prefix.substring(0, prefix.length() - 1) : prefix;


        if (state.containsKey(trimmedPrefix)) {
            try {
                Object value = f.get(last);
                T newObject = (T) state.get(trimmedPrefix);

                // handle lists
                if (value instanceof List<?> list) {
                    ((List<T>) list).add(Integer.parseInt(Arrays.stream(trimmedPrefix.split("\\.")).toList().getLast()), newObject);
                }

                // handle arrays
                else if (value.getClass().isArray())
                    Array.set(value, Integer.parseInt(Arrays.stream(trimmedPrefix.split("\\.")).toList().getLast()), newObject);

                    // handle maps
                else if (value instanceof Map<?, ?> map) {
                    HashMap<String, Object> clone = (HashMap<String, Object>) new HashMap<>(map);
                    clone.put(Arrays.stream(trimmedPrefix.split("\\.")).toList().getLast(), state.get(trimmedPrefix));
                    f.set(last, clone);
                }

                // handle normal values
                else
                    f.set(last, newObject);

                // System.out.println(trimmedPrefix + " @ " + f.getName() + " @ " + last.getClass().getSimpleName() + " = " + newObject);
            } catch (Exception e) {
                if (jNN.LOG_STATE_GENERATION_ERROR)
                    System.err.println("Error while setting " + trimmedPrefix + ": " + e);
            }
            return;
        }

        // iterate over vars
        String allKeys = String.join("\n", state.keySet());
        for (Field field : object.getClass().getDeclaredFields()) {
            try {
                if (Modifier.isPrivate(field.getModifiers())) continue;
                if (Modifier.isTransient(field.getModifiers())) continue;
                if (Modifier.isProtected(field.getModifiers())) continue;
                field.setAccessible(true);

                String name = field.getName();
                Object value = field.get(object);

                // return if key doesn't exist in state
                if (!allKeys.contains(prefix + name))
                    continue;

                // lists
                if (value instanceof List<?> list)
                    for (int i = 0; i < list.size(); i++)
                        loadStateRecursively(list.get(i), object, field, state, prefix + name + "." + i + ".");

                // arrays
                if (field.getType().isArray())
                    for (int i = 0; i < Array.getLength(value); i++)
                        loadStateRecursively(Array.get(value, i), object, field, state, prefix + name + "." + i + ".");

                // maps
                if (value instanceof Map<?, ?> map)
                    map.forEach((key, val) ->
                            loadStateRecursively(val, object, field, state, prefix + name + "." + key + "."));

                else
                    loadStateRecursively(value, object, field, state, prefix + name + ".");
            } catch (Exception e) {
                if (jNN.LOG_STATE_GENERATION_ERROR) {
                    System.err.println("WARNING: Error while loading state. " + e);
                    throw new RuntimeException(e);
                }
            }
        }
    }

    /**
     * Converts a module state to JSON
     * @param state The state to convert
     * @param pretty Pretty print enabled
     */
    public static String stateToJSON(Map<String, Object> state, boolean pretty) {
        GsonBuilder builder = new GsonBuilder();

        if (pretty)
            builder = builder.setPrettyPrinting();

        return SerializationUtils.stateToJSON(state, builder.create());
    }

    /**
     * Converts a module state to JSON
     * @param state The state to convert
     * @param gson The GSON instance to serialize
     */
    public static String stateToJSON(Map<String, Object> state, Gson gson) {
        Map<String, Object> stateCopy = new HashMap<>(state);

        state.forEach((k, v) -> {
            // manually overwrite Tensor
            if (state.get(k) instanceof Tensor<?> tensor)
                stateCopy.put(k, new HashMap<>(){{
                    put("dtype", tensor.dtype.clazz.getName());
                    put("shape", tensor.shape.dimensions);
                    put("data", tensor.data);
                }});

            // manually overwrite DType
            if (state.get(k) instanceof DType<?> dtype)
                stateCopy.put(k, "DType:" + dtype.clazz.getName());
        });

        return gson.toJson(stateCopy);
    }

    /**
     * Construct a model state from it's json representation
     * @param json The json string
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> stateFromJSON(String json) {
        Gson gson = new GsonBuilder().setStrictness(Strictness.STRICT).create();

        Map<String, Object> raw = gson.fromJson(json, new TypeToken<Map<?, ?>>(){}.getType());
        Map<String, Object> state = new HashMap<>(raw);

        raw.forEach((k, v) -> {
            if (v instanceof Map<?,?> m && m.containsKey("shape") && m.containsKey("data") && m.containsKey("dtype")) {
                DType<?> dtype;
                try {
                    dtype = new DType<>(Class.forName(m.get("dtype").toString()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                Tensor<?> tensor = Tensor.of(((ArrayList<?>) m.get("data")).toArray())
                        .reshape(((ArrayList<Double>) m.get("shape")).stream().map(Double::intValue).toArray(Integer[]::new))
                        .asDType(dtype);

                state.put(k, tensor);
            }

            if (v instanceof String str) {
                if (str.startsWith("DType:")) {
                    try {
                        state.put(k, new DType<>(Class.forName(str.split("DType:")[1])));
                    } catch (ClassNotFoundException e) {
                        state.put(k, DType.DEFAULT);
                    }
                }
            }
        });

        return state;
    }

    /**
     * Export a state into a file with pretty print turned off
     * @param state The state to export
     * @param path The path to the file
     */
    public static void export(Map<String, Object> state, String path) {
        export(state, path, false);
    }

    /**
     * Export a state into a file
     * @param state The state to export
     * @param path The path to the file
     * @param pretty Enable pretty print
     */
    @SuppressWarnings("ResultOfMethodCallIgnored")
    public static void export(Map<String, Object> state, String path, boolean pretty) {
        try {
            // create file
            File file = new File(path);
            if (!file.exists()) {
                if (file.getParentFile() != null)
                    file.getParentFile().mkdirs();

                file.createNewFile();
            }

            file.setWritable(true);

            // write json
            Files.writeString(file.toPath(), stateToJSON(state, pretty));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Load a module state from a file
     * @param path The path of the file
     */
    public static Map<String, Object> load(String path) {
        File file = new File(path);

        if (!file.exists()) {
            System.err.println("Trying to load module from non-existing file. Skipping...");
            return null;
        }

        // load state
        try {
            return stateFromJSON(Files.readString(file.toPath()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
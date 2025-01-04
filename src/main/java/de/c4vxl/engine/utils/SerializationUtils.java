package de.c4vxl.engine.utils;

import de.c4vxl.engine.module.Module;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SerializationUtils {
    /**
     * Generate a state of an object
     * @param object The object
     * @param state A map to generate the state into
     * @param prefix A prefix for all paths (leave as "")
     */
    public static void generateStateRecursively(Object object, Map<String, Object> state, String prefix) {
        try {
            if (object != null && (object instanceof de.c4vxl.engine.module.Module ||
                    object instanceof List<?> ||
                    object instanceof Map<?,?> ||
                    object.getClass().isArray())) {

                for (Field field : object.getClass().getDeclaredFields()) {
                    if (Modifier.isPrivate(field.getModifiers())) continue;

                    String name = field.getName();
                    Object value = field.get(object);

                    // submodules
                    if (Module.class.isAssignableFrom(field.getType()))
                        generateStateRecursively(value, state, prefix + name + ".");

                        // lists
                    else if (List.class.isAssignableFrom(field.getType()))
                        for (int i = 0; i < ((List<?>) value).size(); i++)
                            generateStateRecursively(((List<?>) value).get(i), state, prefix + name + "." + i + ".");

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
                System.out.println("Error while setting " + trimmedPrefix + ": " + e);
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
                    return;

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
                    loadStateRecursively(value, object, field, state, prefix + name);
            } catch (Exception e) {
                System.err.println("WARNING: Error while loading state. " + e);
                e.printStackTrace();
            }
        }
    }
}
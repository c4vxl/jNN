package de.c4vxl.engine.module;

import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.security.AnyTypePermission;

import java.io.*;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class serves as a base for creating any kind of modules.
 * A module adds functionality for converting all of its weights into a map which can be saved in a file.
 *
 * @author c4vxl
 */
@SuppressWarnings({"ResultOfMethodCallIgnored", "unchecked"})
public abstract class Module {
    /**
     * Export all the module's parameters into a list
     */
    public Map<String, Object> state() {
        Map<String, Object> state = new HashMap<>();

        for (Field field : this.getClass().getDeclaredFields()) {
            if (field.isSynthetic()) continue;

            field.setAccessible(true);

            // skip private variables
            if (Modifier.isPrivate(field.getModifiers())) continue;

            try {
                Object value = field.get(this);

                // convert modules to states
                if (value instanceof Module module)
                    state.put(field.getName(), module.state());

                // convert lists to keys
                else if (value instanceof List<?> list) {
                    for (int i = 0; i < list.size(); i++) {
                        Object obj = list.get(i);
                        state.put(field.getName() + "." + i, obj instanceof Module ? ((Module) obj).state() : obj);
                    }
                }

                // convert normal values
                else {
                    // copy value if possible
                    try {
                        value = value.getClass().getMethod("clone", String.class).invoke(null);
                    } catch (Exception ignored) {}

                    state.put(field.getName(), value);
                }
            } catch (IllegalAccessException e) {
                System.err.println("Error while trying to create the state! " + e);
            }
        }

        return state;
    }

    /**
     * Load a module's parameters from its state
     * @param state The list of parameters
     */
    public Module load_state(Map<String, Object> state) {
        for (Field field : this.getClass().getDeclaredFields()) {
            field.setAccessible(true);

            // skip private variables
            if (Modifier.isPrivate(field.getModifiers())) continue;

            try {
                Object value = state.get(field.getName());

                if (Module.class.isAssignableFrom(field.getType())) {
                    Module module = (Module) field.get(this);
                    if (module != null && value instanceof Map<?, ?>) {
                        module.load_state((Map<String, Object>) value);
                    }
                } else if (value != null)
                    field.set(this, value);
            } catch (IllegalAccessException e) {
                System.err.println("Error while trying to load the state! " + e);
            }
        }

        return this;
    }

    /**
     * Export this module into a file
     * @param path Path to file
     */
    public Module export(String path) {
        try {
            PrintWriter writer = new PrintWriter(new FileWriter(path));
            XStream stream = new XStream();
            String encoded = stream.toXML(this);
            writer.print(encoded);
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return this;
    }

    /**
     * Load the module from a file
     * @param path Path to file
     */
    public static <T extends Module> T load(String path) {
        File file = new File(path);
        if (!file.exists()) return null;

        file.setReadable(true);

        XStream stream = new XStream();
        stream.addPermission(AnyTypePermission.ANY);

        return (T) stream.fromXML(file);
    }
}
package de.c4vxl.engine.module;

import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.security.AnyTypePermission;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
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
            field.setAccessible(true);

            // skip private variables
            if (Modifier.isPrivate(field.getModifiers())) continue;

            try {
                Object value = field.get(this);

                // copy value if possible
                try {
                    value = value.getClass().getMethod("clone", String.class).invoke(null);
                } catch (Exception ignored) {}

                state.put(field.getName(), value);
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

            try {
                field.set(this, state.get(field.getName()));
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
            String encoded = stream.toXML(state());
            writer.print(encoded);
            writer.close();
        } catch (IOException e) {
            System.err.println("Something went wrong when saving a module to a file!" + e);
        }

        return this;
    }

    /**
     * Load the module from a file
     * @param path Path to file
     */
    public Module load(String path) {
        File file = new File(path);
        if (!file.exists()) return this;

        file.setReadable(true);

        XStream stream = new XStream();
        stream.addPermission(AnyTypePermission.ANY);

        Map<String, Object> loadedState = (Map<String, Object>) stream.fromXML(file);
        load_state(loadedState);

        return this;
    }
}
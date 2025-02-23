package de.c4vxl.engine.module;

import com.thoughtworks.xstream.XStream;
import de.c4vxl.engine.utils.SerializationUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

public class Module {
    /**
     * Captures important parameters in a map
     */
    public Map<String, Object> state() {
        Map<String, Object> state = new LinkedHashMap<>();
        SerializationUtils.generateStateRecursively(this, state, "");
        return state;
    }

    /**
     * Load the modules parameters from a state
     * @param state The state to load from
     */
    @SuppressWarnings("unchecked")
    public <T extends Module> T load_state(Map<String, Object> state) {
        SerializationUtils.loadStateRecursively(this, null, null, state, "");

        return (T) this;
    }

    /**
     * Convert this module into it's xml representation
     */
    public String asXML() {
        XStream xStream = new XStream();
        xStream.setMode(XStream.NO_REFERENCES);
        return xStream.toXML(this);
    }

    /**
     * Load this module from its json representation
     * @param json The json string
     */
    public Module fromJSON(String json) {
        Map<String, Object> state = SerializationUtils.stateFromJSON(json);
        return this.load_state(state);
    }

    /**
     * Export this module as json with pretty = false
     */
    public String asJSON() { return SerializationUtils.stateToJSON(this.state(), false); }

    /**
     * Export this module as json
     * @param pretty Enable pretty print
     */
    public String asJSON(boolean pretty) {
        return SerializationUtils.stateToJSON(this.state(), pretty);
    }

    /**
     * Load this modules state from a xml string
     * @param xml The XML
     */
    @SuppressWarnings("unchecked")
    public static <T extends Module> T fromXML(String xml) {
        XStream xStream = new XStream();
        xStream.setMode(XStream.NO_REFERENCES);
        xStream.allowTypesByWildcard(new String[] {
                "de.c4vxl.**",
                "java.lang.**",
                "java.util.**"
        });
        return (T) xStream.fromXML(xml);
    }

    /**
     * Export this module into a file
     * @param path The path to the file
     */
    public Module export(String path) {
        SerializationUtils.export(this.state(), path);
        return this;
    }

    /**
     * Load this module from a file
     * @param path The path to the file
     */
    public static <T extends Module> T load(String path) {
        File file = new File(path);

        if (!file.exists()) {
            System.err.println("Trying to load module from non-existing file. Skipping...");
            return null;
        }

        // load from file
        try {
            return Module.fromXML(Files.readString(file.toPath()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Load this module from a file
     * @param path The path to the file
     * @param type The module type
     */
    @SuppressWarnings("unchecked")
    public static <T> T load(String path, Class<T> type) { return (T) Module.load(path); }

    /**
     * Try to find and invoke a forward function
     * @param args The arguments to pass to the forward function
     */
    public Object _forward(Object... args) {
        try {
            return this.getClass()
                    .getMethod("forward", Arrays.stream(args).map(Object::getClass).toArray(Class[]::new))
                    .invoke(this, args);
        } catch (Exception e) {
            System.err.println("No method for forwarding found! (Module: " + this.getClass().getName() + ")");
            return null;
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + this.state();
    }
}
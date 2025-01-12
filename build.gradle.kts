plugins {
    id("java")
}

group = "de.c4vxl"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.thoughtworks.xstream:xstream:1.4.21")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}
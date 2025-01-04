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
    implementation("org.codehaus.jettison:jettison:1.5.4")
}
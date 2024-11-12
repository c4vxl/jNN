plugins {
    id("java")
}

group = "de.c4vxl"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    implementation("com.thoughtworks.xstream:xstream:1.4.20") // used for saving modules
}

tasks.test {
    useJUnitPlatform()
}
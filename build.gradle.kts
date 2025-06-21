plugins {
    id("java")
    `maven-publish`
}

group = "de.c4vxl"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.google.code.gson:gson:2.12.1")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
}

tasks.jar {
    exclude("de/c4vxl/Main.class")
}

java {
    withSourcesJar()
    withJavadocJar()
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])

            groupId = "de.c4vxl"
            artifactId = "jNN"
            version = "1.0.0"

            artifact(tasks.named("sourcesJar"))
            artifact(tasks.named("javadocJar"))
        }
    }

    repositories {
        maven(layout.buildDirectory.dir("repo"))
    }
}
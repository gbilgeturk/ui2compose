// Android library holding the generated screens (src/main/java/generated/sNN)
// and one Paparazzi snapshot test per screen (src/test/java/generated/sNN).
// Both source trees are written by harness/compile_and_render.py.
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.paparazzi)
}

android {
    namespace = "ui2compose.harness.screens"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    testImplementation(libs.junit)
}

tasks.withType<Test>().configureEach {
    maxHeapSize = "2g"
    // The snapshot images are the product of the run: never skip it as up to date.
    outputs.upToDateWhen { false }
}

// Prints the Compose artifact versions the BOM resolved to, so that the
// report records what the code was really compiled against.
tasks.register("printResolvedVersions") {
    val runtimeClasspath = configurations.named("debugRuntimeClasspath")
    doLast {
        val wanted = setOf(
            "androidx.compose.material3:material3",
            "androidx.compose.ui:ui",
            "androidx.compose.foundation:foundation",
            "androidx.compose.runtime:runtime",
        )
        runtimeClasspath.get().incoming.resolutionResult.allComponents
            .mapNotNull { it.moduleVersion }
            .filter { "${it.group}:${it.name}" in wanted }
            .sortedBy { "${it.group}:${it.name}" }
            .forEach { println("RESOLVED ${it.group}:${it.name}=${it.version}") }
    }
}

package com.z32pro.z32lite_app

import android.content.Context
import android.util.Log

class LlamaAndroid {
    companion object {
        init {
            // Load the native library
            try {
                System.loadLibrary("z32lite")
            } catch (e: Exception) {
                Log.e("LlamaAndroid", "Could not load z32lite native library", e)
            }
        }
    }

    private var modelHandle: Long = 0

    // JNI Methods
    external fun loadModel(modelPath: String, nThreads: Int, nCtx: Int): Long
    external fun generate(modelHandle: Long, systemPrompt: String, userPrompt: String, maxTokens: Int, temperature: Float): String
    external fun unloadModel(modelHandle: Long)

    fun load(path: String, threads: Int, ctx: Int): Boolean {
        modelHandle = loadModel(path, threads, ctx)
        return modelHandle != 0L
    }

    fun infer(system: String, user: String, maxTokens: Int, temp: Float): String {
        if (modelHandle == 0L) return "Error: Model not loaded"
        return generate(modelHandle, system, user, maxTokens, temp)
    }

    fun unload() {
        if (modelHandle != 0L) {
            unloadModel(modelHandle)
            modelHandle = 0
        }
    }
}

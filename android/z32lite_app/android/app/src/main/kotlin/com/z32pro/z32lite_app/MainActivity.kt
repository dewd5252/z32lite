package com.z32pro.z32lite_app

import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.media.session.MediaSessionManager
import android.net.Uri
import android.provider.AlarmClock
import android.provider.ContactsContract
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : FlutterActivity() {

    private val SYSTEM_CHANNEL = "com.z32pro.z32lite/system"
    private val MODEL_CHANNEL  = "com.z32pro.z32lite/model"

    // -------------------------------------------------------
    // FlutterEngine: register both MethodChannels
    // -------------------------------------------------------
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // SYSTEM CHANNEL: handles volume, media, search, contacts, flashlight, alarm
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, SYSTEM_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "setVolume"      -> handleSetVolume(call.arguments as Map<*, *>, result)
                    "mediaControl"   -> handleMediaControl(call.arguments as Map<*, *>, result)
                    "searchContacts" -> handleSearchContacts(call.arguments as Map<*, *>, result)
                    "flashlight"     -> handleFlashlight(call.arguments as Map<*, *>, result)
                    "setAlarm"       -> handleSetAlarm(call.arguments as Map<*, *>, result)
                    else             -> result.notImplemented()
                }
            }

        // MODEL CHANNEL: stub for llama.cpp JNI integration
        // Full implementation needs llama.cpp compiled as .so + JNI wrappers
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, MODEL_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "loadModel"   -> handleLoadModel(call.arguments as Map<*, *>, result)
                    "generate"    -> handleGenerate(call.arguments as Map<*, *>, result)
                    "unloadModel" -> handleUnloadModel(result)
                    else          -> result.notImplemented()
                }
            }
    }

    // -------------------------------------------------------
    // VOLUME CONTROL
    // -------------------------------------------------------
    private fun handleSetVolume(args: Map<*, *>, result: MethodChannel.Result) {
        val direction = args["direction"] as? String ?: "up"
        val stream    = when (args["stream"] as? String) {
            "media"        -> AudioManager.STREAM_MUSIC
            "notification" -> AudioManager.STREAM_NOTIFICATION
            "mute"         -> AudioManager.STREAM_RING
            else           -> AudioManager.STREAM_RING
        }
        val audio = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val flag  = AudioManager.FLAG_SHOW_UI

        when (direction) {
            "up"   -> audio.adjustStreamVolume(stream, AudioManager.ADJUST_RAISE, flag)
            "down" -> audio.adjustStreamVolume(stream, AudioManager.ADJUST_LOWER, flag)
            "mute" -> audio.adjustStreamVolume(stream, AudioManager.ADJUST_MUTE,  flag)
        }
        result.success(true)
    }

    // -------------------------------------------------------
    // MEDIA CONTROL (via AudioManager key events)
    // -------------------------------------------------------
    private fun handleMediaControl(args: Map<*, *>, result: MethodChannel.Result) {
        val action  = args["action"] as? String ?: ""
        val keyCode = when (action) {
            "media_next_track" -> android.view.KeyEvent.KEYCODE_MEDIA_NEXT
            "media_prev_track" -> android.view.KeyEvent.KEYCODE_MEDIA_PREVIOUS
            "media_play"       -> android.view.KeyEvent.KEYCODE_MEDIA_PLAY
            "media_pause"      -> android.view.KeyEvent.KEYCODE_MEDIA_PAUSE
            else               -> -1
        }
        if (keyCode == -1) { result.success(false); return }

        val audio = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audio.dispatchMediaKeyEvent(android.view.KeyEvent(android.view.KeyEvent.ACTION_DOWN, keyCode))
        audio.dispatchMediaKeyEvent(android.view.KeyEvent(android.view.KeyEvent.ACTION_UP,   keyCode))
        result.success(true)
    }

    // -------------------------------------------------------
    // CONTACTS SEARCH
    // -------------------------------------------------------
    private fun handleSearchContacts(args: Map<*, *>, result: MethodChannel.Result) {
        val query  = args["query"] as? String ?: ""
        val intent = Intent(Intent.ACTION_SEARCH).apply {
            setPackage("com.android.contacts")
            putExtra(android.app.SearchManager.QUERY, query)
        }
        startActivity(intent)
        result.success("🔍 بفتح لك جهات الاتصال...")
    }

    // -------------------------------------------------------
    // FLASHLIGHT
    // -------------------------------------------------------
    private fun handleFlashlight(args: Map<*, *>, result: MethodChannel.Result) {
        val state   = args["state"] as? Boolean ?: false
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            val cameraId = manager.cameraIdList[0]
            manager.setTorchMode(cameraId, state)
            result.success(true)
        } catch (e: Exception) {
            result.error("FLASHLIGHT_ERROR", e.message, null)
        }
    }

    // -------------------------------------------------------
    // SET ALARM
    // -------------------------------------------------------
    private fun handleSetAlarm(args: Map<*, *>, result: MethodChannel.Result) {
        val time  = (args["time"] as? String ?: "07:00").split(":")
        val label = args["label"] as? String ?: "Z32LITE"
        val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
            putExtra(AlarmClock.EXTRA_HOUR,    time[0].toIntOrNull() ?: 7)
            putExtra(AlarmClock.EXTRA_MINUTES, time.getOrNull(1)?.toIntOrNull() ?: 0)
            putExtra(AlarmClock.EXTRA_MESSAGE, label)
            putExtra(AlarmClock.EXTRA_SKIP_UI, true) // Set without opening app
        }
        startActivity(intent)
        result.success(true)
    }

    // -------------------------------------------------------
    // MODEL: llama.cpp JNI integration
    // -------------------------------------------------------
    private val llama = LlamaAndroid()
    private var modelLoaded = false

    private fun handleLoadModel(args: Map<*, *>, result: MethodChannel.Result) {
        val path = args["path"] as? String ?: ""
        val threads = (args["threads"] as? Int) ?: 4
        val ctx = (args["contextSize"] as? Int) ?: 2048
        
        modelLoaded = llama.load(path, threads, ctx)
        result.success(modelLoaded)
    }

    private fun handleGenerate(args: Map<*, *>, result: MethodChannel.Result) {
        if (!modelLoaded) {
            result.error("MODEL_NOT_LOADED", "Load the model first", null)
            return
        }
        val system = args["system"] as? String ?: ""
        val user   = args["user"]   as? String ?: ""
        val maxTokens = (args["maxTokens"] as? Int) ?: 256
        val temp = (args["temperature"] as? Double)?.toFloat() ?: 0.7f

        CoroutineScope(Dispatchers.IO).launch {
            val response = llama.infer(system, user, maxTokens, temp)
            runOnUiThread { result.success(response) }
        }
    }

    private fun handleUnloadModel(result: MethodChannel.Result) {
        llama.unload()
        modelLoaded = false
        result.success(true)
    }
}

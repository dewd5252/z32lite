/// model_manager.dart
/// Manages loading and running the GGUF model via llama.cpp FFI.
///
/// NOTE: This uses a MethodChannel bridge to native Kotlin/JNI because
/// llama.cpp's Android bindings require native code. The actual llama.cpp
/// JNI setup is in android/app/src/main/kotlin/...
library;

import 'package:flutter/services.dart';

enum ModelStatus { notLoaded, loading, ready, error }

class ModelManager {
  static const _channel = MethodChannel('com.z32pro.z32lite/model');
  static final ModelManager _instance = ModelManager._internal();
  factory ModelManager() => _instance;
  ModelManager._internal();

  ModelStatus _status = ModelStatus.notLoaded;
  ModelStatus get status => _status;
  bool get isReady => _status == ModelStatus.ready;

  /// Load the GGUF model from the given file path.
  /// Call this once at app startup from a background isolate.
  Future<bool> loadModel(String ggufPath) async {
    _status = ModelStatus.loading;
    try {
      final success = await _channel.invokeMethod<bool>('loadModel', {
        'path': ggufPath,
        'threads': 4, // CPU threads for inference
        'contextSize': 2048, // Context window
        'batchSize': 512,
      });
      _status = success == true ? ModelStatus.ready : ModelStatus.error;
      return _status == ModelStatus.ready;
    } on PlatformException {
      _status = ModelStatus.error;
      return false;
    }
  }

  /// Generate a response from the model given a user message.
  /// [onToken] is called for each generated token (streaming).
  Future<String> generate({
    required String userMessage,
    String? systemPrompt,
    int maxTokens = 256,
    double temperature = 0.7,
    void Function(String token)? onToken,
  }) async {
    if (!isReady) return 'الموديل مش محمل بعد! جرب تانى.';

    final system =
        systemPrompt ??
        'أنت Z32LITE، مساعد ذكاء اصطناعي خفيف وسريع. '
            'تتحدث العربية والإنجليزية وتفهم اللهجة المصرية. '
            'عند الحاجة لتنفيذ أمر جهاز: SYSTEM_ACTION:{json}. '
            'عند الحاجة للبحث: SYSTEM_ACTION:{"action":"search_web","query":"..."}. '
            'عند الحاجة لموافقة المستخدم: NOTIFY_USER:{"message":"...", "action_pending":"..."}.';

    try {
      // Stream tokens back via event channel (EventChannel) in production.
      // For simplicity, we use a single call here.
      final result = await _channel.invokeMethod<String>('generate', {
        'system': system,
        'user': userMessage,
        'maxTokens': maxTokens,
        'temperature': temperature,
      });
      return result ?? '...';
    } on PlatformException catch (e) {
      return '❌ خطأ في الموديل: ${e.message}';
    }
  }

  /// Release model from memory.
  Future<void> unload() async {
    await _channel.invokeMethod('unloadModel');
    _status = ModelStatus.notLoaded;
  }
}

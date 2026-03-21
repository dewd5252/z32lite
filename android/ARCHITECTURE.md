# Z32LITE Android Architecture

## Tech Stack
- **Language:** Flutter (Dart) - cross-platform + rich UI
- **Model Runtime:** llama.cpp (via FFI bindings) - runs GGUF directly
- **Target:** Android 8.0+ (API 26+), 2-3GB RAM devices

## App Flow

```
User Voice/Text Input
        ↓
  Z32LITE Model (on-device, GGUF)
        ↓
  Response Parser
  ┌─────┴─────────────────┐
  │                       │
Normal Text         SYSTEM_ACTION / NOTIFY_USER
  │                       │
Display            Action Handler
                   ┌──────┴──────────────────────┐
                   │           │                 │
             set_volume   media_control     search_web
             flashlight   set_alarm         (Chrome Tab)
```

## Permissions Required
| Permission | Purpose |
|---|---|
| `BIND_ACCESSIBILITY_SERVICE` | Control volume and media |
| `RECEIVE_BOOT_COMPLETED` | Re-start listener on reboot |
| `FOREGROUND_SERVICE` | Keep model loaded in background |
| `POST_NOTIFICATIONS` | NOTIFY_USER alerts |
| `INTERNET` | (Only for search_web via browser) |

## Key Modules
1. **ModelManager** - Load/unload GGUF model via llama.cpp FFI
2. **ActionParser** - Parse `SYSTEM_ACTION:{...}` JSON from model output
3. **SystemController** - Execute parsed actions (volume, media, etc.)
4. **SearchModule** - Launch Chrome Custom Tabs for web search
5. **NotificationManager** - Show `NOTIFY_USER` alerts to user

## GGUF Integration (llama.cpp FFI)
```dart
// Load model
final model = await LlamaModel.load('assets/z32lite_Q4_K_M.gguf');

// Inference
final response = await model.generate(
  prompt: buildPrompt(systemPrompt, userInput),
  maxTokens: 256,
  temperature: 0.7,
);

// Parse action
if (response.startsWith('SYSTEM_ACTION:')) {
  final json = jsonDecode(response.substring(14));
  await SystemController.execute(json);
}
```

## Monetization (Phase 3)
- **Free:** 1.5B model, basic features
- **Pro (subscription):** 3B model option, priority inference, cloud backup of preferences

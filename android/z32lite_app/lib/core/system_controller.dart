/// system_controller.dart
/// Executes parsed system actions by calling Android APIs via MethodChannel.
library;

import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';
import 'action_parser.dart';

class SystemController {
  // Method channel to communicate with native Android Kotlin code
  static const _channel = MethodChannel('com.z32pro.z32lite/system');

  /// Execute a parsed action. Returns a user-friendly result message.
  static Future<String> execute(ParsedAction action) async {
    switch (action.type) {
      case ActionType.setVolume:
        return _setVolume(action.params);
      case ActionType.mediaControl:
        return _mediaControl(action.params);
      case ActionType.searchWeb:
        return _searchWeb(action.params);
      case ActionType.searchContacts:
        return _searchContacts(action.params);
      case ActionType.flashlight:
        return _flashlight(action.params);
      case ActionType.setAlarm:
        return _setAlarm(action.params);
      case ActionType.notifyUser:
        return action.rawText; // Handled by NotificationManager
      case ActionType.none:
        return action.rawText;
    }
  }

  // ------- Volume -------
  static Future<String> _setVolume(Map<String, dynamic> params) async {
    final direction = params['direction'] ?? 'up';
    final stream = params['stream'] ?? 'ring'; // ring | media | notification
    try {
      await _channel.invokeMethod('setVolume', {
        'direction': direction,
        'stream': stream,
      });
      return direction == 'up'
          ? '🔊 رفعت الصوت!'
          : direction == 'mute'
          ? '🔇 الصوت اتكتم!'
          : '🔉 خفضت الصوت!';
    } on PlatformException catch (e) {
      return '❌ مقدرتش أتحكم في الصوت: ${e.message}';
    }
  }

  // ------- Media -------
  static Future<String> _mediaControl(Map<String, dynamic> params) async {
    final action = params['action'] as String;
    try {
      await _channel.invokeMethod('mediaControl', {'action': action});
      const messages = {
        'media_next_track': '⏭️ غيرت الأغنية!',
        'media_prev_track': '⏮️ رجعت للأغنية اللي فاتت!',
        'media_play': '▶️ شغلت الموسيقى!',
        'media_pause': '⏸️ وقفت الموسيقى!',
      };
      return messages[action] ?? '✅ تم!';
    } on PlatformException catch (e) {
      return '❌ خطأ في التحكم في الميديا: ${e.message}';
    }
  }

  // ------- Web Search (Zero-cost via Chrome Custom Tabs) -------
  static Future<String> _searchWeb(Map<String, dynamic> params) async {
    final query = params['query']?.toString() ?? '';
    if (query.isEmpty) return '⚠️ محتاج تقولي بتدور على إيه!';

    final encoded = Uri.encodeComponent(query);
    final uri = Uri.parse('https://www.google.com/search?q=$encoded');

    if (await canLaunchUrl(uri)) {
      // LaunchMode.externalApplication = Chrome Custom Tabs on Android
      await launchUrl(uri, mode: LaunchMode.externalApplication);
      return '🔍 بفتحلك جوجل بـ "$query"...';
    } else {
      return '❌ مقدرتش أفتح المتصفح!';
    }
  }

  // ------- Contacts Search -------
  static Future<String> _searchContacts(Map<String, dynamic> params) async {
    final query = params['query']?.toString() ?? '';
    try {
      final result = await _channel.invokeMethod<String>('searchContacts', {
        'query': query,
      });
      return result ?? '📞 مش لاقي نتايج لـ "$query"';
    } on PlatformException catch (e) {
      return '❌ خطأ في البحث في جهات الاتصال: ${e.message}';
    }
  }

  // ------- Flashlight -------
  static Future<String> _flashlight(Map<String, dynamic> params) async {
    final state = params['state'] == 'on';
    try {
      await _channel.invokeMethod('flashlight', {'state': state});
      return state ? '🔦 وديت التورش!' : '🔦 أفلت التورش!';
    } on PlatformException catch (e) {
      return '❌ خطأ في التورش: ${e.message}';
    }
  }

  // ------- Set Alarm -------
  static Future<String> _setAlarm(Map<String, dynamic> params) async {
    final time = params['time']?.toString() ?? '07:00';
    final label = params['label']?.toString() ?? 'Z32LITE';
    try {
      await _channel.invokeMethod('setAlarm', {'time': time, 'label': label});
      return '⏰ ظبطت المنبه على الساعة $time!';
    } on PlatformException catch (e) {
      return '❌ خطأ في المنبه: ${e.message}';
    }
  }
}

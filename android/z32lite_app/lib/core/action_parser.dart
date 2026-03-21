/// action_parser.dart
/// Parses Z32LITE model output and extracts structured system actions.
/// Model outputs SYSTEM_ACTION:{json} or NOTIFY_USER:{json}
library;

import 'dart:convert';

enum ActionType {
  none,
  setVolume,
  mediaControl,
  searchWeb,
  searchContacts,
  flashlight,
  setAlarm,
  notifyUser,
}

class ParsedAction {
  final ActionType type;
  final Map<String, dynamic> params;
  final String rawText; // The user-facing text part (if any)

  const ParsedAction({
    required this.type,
    required this.params,
    required this.rawText,
  });

  bool get hasAction => type != ActionType.none;

  @override
  String toString() => 'ParsedAction(type: $type, params: $params)';
}

class ActionParser {
  static const _systemPrefix = 'SYSTEM_ACTION:';
  static const _notifyPrefix = 'NOTIFY_USER:';

  /// Parse the raw model output and return a [ParsedAction].
  static ParsedAction parse(String modelOutput) {
    final trimmed = modelOutput.trim();

    // Check for NOTIFY_USER
    if (trimmed.startsWith(_notifyPrefix)) {
      final jsonStr = trimmed.substring(_notifyPrefix.length).trim();
      try {
        final json = jsonDecode(jsonStr) as Map<String, dynamic>;
        return ParsedAction(
          type: ActionType.notifyUser,
          params: json,
          rawText: json['message']?.toString() ?? '',
        );
      } catch (_) {
        return ParsedAction(type: ActionType.none, params: {}, rawText: trimmed);
      }
    }

    // Check for SYSTEM_ACTION
    if (trimmed.startsWith(_systemPrefix)) {
      final jsonStr = trimmed.substring(_systemPrefix.length).trim();
      try {
        final json = jsonDecode(jsonStr) as Map<String, dynamic>;
        final action = json['action']?.toString() ?? '';
        return ParsedAction(
          type: _mapAction(action),
          params: json,
          rawText: '',
        );
      } catch (_) {
        return ParsedAction(type: ActionType.none, params: {}, rawText: trimmed);
      }
    }

    // Normal conversational text
    return ParsedAction(type: ActionType.none, params: {}, rawText: trimmed);
  }

  static ActionType _mapAction(String action) {
    switch (action) {
      case 'set_volume':
        return ActionType.setVolume;
      case 'media_next_track':
      case 'media_prev_track':
      case 'media_play':
      case 'media_pause':
        return ActionType.mediaControl;
      case 'search_web':
        return ActionType.searchWeb;
      case 'search_contacts':
        return ActionType.searchContacts;
      case 'flashlight':
        return ActionType.flashlight;
      case 'set_alarm':
        return ActionType.setAlarm;
      default:
        return ActionType.none;
    }
  }
}

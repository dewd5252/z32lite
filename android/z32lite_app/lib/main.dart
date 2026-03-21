/// main.dart - Z32LITE entry point
library;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'ui/chat_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Force portrait orientation
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  // Dark system UI
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF0A0E1A),
    ),
  );
  runApp(const Z32LiteApp());
}

class Z32LiteApp extends StatelessWidget {
  const Z32LiteApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Z32LITE',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0A0E1A),
        colorScheme: ColorScheme.dark(
          primary: const Color(0xFF6C63FF),
          secondary: const Color(0xFF3B82F6),
          surface: const Color(0xFF111827),
        ),
        textTheme: const TextTheme(
          bodyMedium: TextStyle(fontFamily: 'Cairo', color: Colors.white),
        ),
      ),
      home: const ChatScreen(),
    );
  }
}

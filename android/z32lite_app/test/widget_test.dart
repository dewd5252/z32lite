import 'package:flutter_test/flutter_test.dart';
import 'package:z32lite_app/main.dart';

void main() {
  testWidgets('Z32LiteApp smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const Z32LiteApp());

    // Verify that the welcome message is shown.
    expect(find.text('أهلاً بيك في Z32LITE'), findsOneWidget);
  });
}

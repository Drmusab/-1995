"""
Life Organizer Demo - عرض توضيحي لمنظم الحياة
Author: Drmusab
Last Modified: 2025-01-20

Demonstrates the Life Organizer skill functionality including:
- Arabic voice commands for Kanban board management
- Mood and energy tracking integration
- Adaptive recommendations based on user state
- Goal decomposition and daily planning
"""

import asyncio
import json
from datetime import datetime, timezone


def print_header(title_ar: str, title_en: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"🌟 {title_ar} | {title_en}")
    print("=" * 80)


def print_section(title: str):
    """Print a section header."""
    print(f"\n📋 {title}")
    print("-" * 50)


def simulate_voice_input(command: str, language: str = "ar"):
    """Simulate voice input processing."""
    print(f"🎤 {command}")


def simulate_response(response: str):
    """Simulate system response."""
    print(f"🤖 {response}")


async def demo_goal_decomposition():
    """Demo goal decomposition feature."""
    print_section("Goal Decomposition - تحليل الأهداف")
    
    # Arabic goal
    print("User Goal (Arabic):")
    simulate_voice_input("أريد تطوير مهاراتي في البرمجة خلال 6 أشهر")
    
    print("\nSystem Analysis:")
    simulate_response("""تم تحليل هدفك 'تطوير مهاراتي في البرمجة خلال 6 أشهر' بنجاح!

📋 خطوات العمل المقترحة:
1. اختيار لغة البرمجة المناسبة
   ⏱️ المدة المقدرة: أسبوع واحد
   🎯 الأولوية: عالي

2. تعلم أساسيات اللغة المختارة
   ⏱️ المدة المقدرة: 4 أسابيع
   🎯 الأولوية: عالي

3. بناء مشروع تطبيقي صغير
   ⏱️ المدة المقدرة: 6 أسابيع
   🎯 الأولوية: متوسط

4. التقدم لمشاريع أكثر تعقيداً
   ⏱️ المدة المقدرة: 8 أسابيع
   🎯 الأولوية: متوسط

💡 هل تريد إضافة هذه المهام إلى لوحة كانبان الصوتية؟""")


async def demo_mood_tracking():
    """Demo mood and energy tracking."""
    print_section("Mood & Energy Tracking - تتبع المزاج والطاقة")
    
    print("Analyzing voice input for mood...")
    simulate_voice_input("أشعر بالحماس والطاقة اليوم! جاهز لإنجاز مهام صعبة")
    
    print("\nAnalyzing facial expression...")
    print("📷 [Facial expression captured: Alert eyes, slight smile, engaged posture]")
    
    print("\nMood & Energy Analysis:")
    simulate_response("""📊 تحليل المزاج والطاقة:

🎭 المزاج الحالي: عالي
⚡ مستوى الطاقة: عالي
🎯 دقة التحليل: 87%
📡 المصادر: voice, vision

💡 التوصيات المناسبة لحالتك:
1. عمل عميق ومركز
   ⏱️ المدة المثلى: 90 دقيقة
   📝 السبب: حالتك جيدة للتركيز العميق

2. حل المشاكل المعقدة
   ⏱️ المدة المثلى: 120 دقيقة
   📝 السبب: طاقتك العالية تساعدك على التركيز في المهام الصعبة

🧠 نصائح ذكية إضافية:
1. وقت مثالي للمهام الصعبة: حالتك ممتازة - اغتنم الفرصة للمهام المعقدة""")


async def demo_voice_kanban():
    """Demo voice-controlled Kanban board."""
    print_section("Voice Kanban Interface - واجهة كانبان الصوتية")
    
    print("Creating tasks with Arabic voice commands:")
    
    # Add tasks
    simulate_voice_input("أضف مهمة كتابة التقرير الشهري")
    simulate_response("تم إنشاء المهمة 'كتابة التقرير الشهري' في عمود 'للقيام' بأولوية 'متوسط'")
    
    simulate_voice_input("أضف مهمة مراجعة الكود عالي الأولوية")
    simulate_response("تم إنشاء المهمة 'مراجعة الكود' في عمود 'للقيام' بأولوية 'عالي'")
    
    simulate_voice_input("أضف مهمة اجتماع الفريق عاجل")
    simulate_response("تم إنشاء المهمة 'اجتماع الفريق' في عمود 'للقيام' بأولوية 'عاجل'")
    
    # Show board
    print("\nViewing the board:")
    simulate_voice_input("أظهر اللوحة")
    simulate_response("""لوحة 'اللوحة الشخصية':

📋 قائمة المهام (0 مهام):

📋 للقيام (3 مهام):
  🔴 اجتماع الفريق
  🟠 مراجعة الكود
  🟡 كتابة التقرير الشهري

📋 قيد التنفيذ (0 مهام):

📋 مراجعة (0 مهام):

📋 مكتمل (0 مهام):""")
    
    # Move tasks
    print("\nMoving tasks:")
    simulate_voice_input("انقل اجتماع الفريق إلى قيد التنفيذ")
    simulate_response("تم نقل 'اجتماع الفريق' من 'للقيام' إلى 'قيد التنفيذ'")
    
    simulate_voice_input("أكملت مراجعة الكود")
    simulate_response("ممتاز! تم تحريك 'مراجعة الكود' إلى عمود المكتمل 🎉")


async def demo_adaptive_scheduling():
    """Demo adaptive scheduling based on mood."""
    print_section("Adaptive Scheduling - الجدولة التكيفية")
    
    print("Creating mood-aware schedule...")
    print("Current state: High mood, High energy")
    
    simulate_response("""📅 تم إنشاء جدولك التكيفي:

⏰ 09:00–11:00: مراجعة الكود (عمل عميق) - 120 دقيقة
☕ 11:00–11:15: استراحة
⏰ 11:15–12:45: كتابة التقرير الشهري - 90 دقيقة
🍽️ 12:45–13:30: استراحة غداء
⏰ 13:30–14:30: اجتماع الفريق - 60 دقيقة

🎭 تم إنشاء الجدول بناءً على حالتك الحالية:
• المزاج: عالي
• الطاقة: عالي

💡 نصائح إضافية:
• وقت مثالي للمهام الصعبة: حالتك ممتازة - اغتنم الفرصة للمهام المعقدة
• استثمر ذروة طاقتك: أنت في أفضل حالاتك - ركز على أهم مشاريعك""")


async def demo_tired_state():
    """Demo system behavior when user is tired."""
    print_section("Adaptive Response to Fatigue - التكيف مع التعب")
    
    print("Detecting tired state from voice and vision...")
    simulate_voice_input("أشعر بالتعب والإرهاق... لا أستطيع التركيز")
    print("📷 [Expression: Droopy eyes, yawning, slouched posture]")
    
    simulate_response("""📊 تحليل المزاج والطاقة:

🎭 المزاج الحالي: منخفض
⚡ مستوى الطاقة: منهك
🎯 دقة التحليل: 92%
📡 المصادر: voice, vision

💡 التوصيات المناسبة لحالتك:
1. راحة واسترخاء
   ⏱️ المدة المثلى: 30 دقيقة
   📝 السبب: تحتاج إلى راحة - خذ استراحة قبل المتابعة

🧠 نصائح ذكية إضافية:
1. حان وقت الراحة: تبدو متعباً جداً. خذ استراحة لمدة 15-30 دقيقة
2. نشط طاقتك: تحرك قليلاً أو اشرب الماء أو تنفس بعمق

🔄 إعادة جدولة المهام الصعبة لوقت أفضل؟""")

    print("\nAdapting schedule for tired state:")
    simulate_voice_input("نعم، أعد جدولة المهام الصعبة")
    
    simulate_response("""📅 جدولك المُحدث (مناسب للحالة الحالية):

⏰ 09:00–09:30: استراحة وتنشيط - 30 دقيقة
⏰ 09:30–10:00: مهام خفيفة (رد على الإيميلات) - 30 دقيقة
☕ 10:00–10:15: استراحة
⏰ 10:15–10:45: مهام بسيطة (تنظيم الملفات) - 30 دقيقة

🔄 تم تأجيل المهام الصعبة:
• مراجعة الكود → مؤجل لبعد الظهر
• كتابة التقرير → مؤجل للغد
• اجتماع الفريق → تم الاحتفاظ بالموعد (أولوية عاجلة)

💡 نصيحة: خذ قسطاً من الراحة أولاً، ثم عاود المحاولة لاحقاً""")


async def demo_english_support():
    """Demo English language support."""
    print_section("English Language Support - دعم اللغة الإنجليزية")
    
    print("Switching to English mode...")
    
    simulate_voice_input("Add task review presentation", "en")
    simulate_response("Created task 'review presentation' in 'To Do' column with 'Medium' priority")
    
    simulate_voice_input("Show board", "en")
    simulate_response("""Board 'Personal Board':

📋 Backlog (0 tasks):

📋 To Do (1 tasks):
  🟡 review presentation

📋 In Progress (1 tasks):
  🔴 team meeting

📋 Review (0 tasks):

📋 Done (1 tasks):
  🟠 code review""")

    simulate_voice_input("I feel energetic and motivated today!", "en")
    simulate_response("""📊 Mood and Energy Analysis:

🎭 Current Mood: High
⚡ Energy Level: High
🎯 Analysis Accuracy: 89%
📡 Sources: voice

💡 Recommendations for your current state:
1. Deep focused work
   ⏱️ Optimal duration: 90 minutes
   📝 Reason: You're in good state for deep focus

🧠 Additional smart tips:
1. Perfect time for challenging tasks: You're in great state - seize the opportunity for complex tasks""")


async def run_demo():
    """Run the complete Life Organizer demo."""
    print_header("Life Organizer Demo", "عرض توضيحي لمنظم الحياة")
    
    print("""
🌟 Welcome to the Life Organizer - أهلاً بك في منظم الحياة الذكي

This demo showcases the comprehensive Life Organizer skill that provides:
يعرض هذا العرض مهارة منظم الحياة الشامل التي تقدم:

✨ Key Features - الميزات الرئيسية:
• 🎯 Goal decomposition into actionable steps - تحليل الأهداف إلى خطوات عملية
• 📊 Mood & energy tracking via voice and vision - تتبع المزاج والطاقة عبر الصوت والرؤية
• 🗣️ Voice-controlled Kanban board - لوحة كانبان صوتية
• 🧠 Adaptive recommendations - توصيات ذكية تكيفية
• 🌍 Arabic-first with English support - العربية أولاً مع دعم الإنجليزية
""")
    
    # Run demo sections
    await demo_goal_decomposition()
    await asyncio.sleep(1)
    
    await demo_mood_tracking()
    await asyncio.sleep(1)
    
    await demo_voice_kanban()
    await asyncio.sleep(1)
    
    await demo_adaptive_scheduling()
    await asyncio.sleep(1)
    
    await demo_tired_state()
    await asyncio.sleep(1)
    
    await demo_english_support()
    
    # Conclusion
    print_header("Demo Complete", "انتهى العرض التوضيحي")
    print("""
🎉 Life Organizer Demo Complete!

✅ Features Demonstrated:
• Goal decomposition with step-by-step planning
• Multimodal mood and energy tracking
• Natural Arabic voice commands for task management
• Intelligent mood-aware scheduling and recommendations
• Adaptive behavior based on user state (energetic vs tired)
• Seamless bilingual support (Arabic/English)

🚀 The Life Organizer is ready to help users:
• Break down complex goals into manageable steps
• Track mood and energy to optimize productivity
• Manage tasks naturally with voice commands
• Get personalized recommendations based on current state
• Plan their day adaptively based on how they feel

💡 Next Steps:
• Integration with the main AI assistant system
• User testing and feedback collection
• Continuous learning from user interactions
• Enhancement of recommendation algorithms

شكراً لمشاهدة العرض! - Thank you for watching the demo!
""")


if __name__ == "__main__":
    asyncio.run(run_demo())
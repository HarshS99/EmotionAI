# test_grok.py
from grok_client import GrokClient
from content_recommender import ContentRecommender

def test_all():
    print("🧪 Testing Grok Integration...\n")
    
    # Test 1: Grok Client
    print("1️⃣ Testing Grok Client...")
    try:
        grok = GrokClient()
        response = grok.chat_completion([
            {"role": "user", "content": "Say hello in one sentence"}
        ])
        print(f"✅ Grok Response: {response}\n")
    except Exception as e:
        print(f"❌ Grok Error: {e}\n")
        return
    
    # Test 2: Content Recommender
    print("2️⃣ Testing Content Recommender...")
    try:
        recommender = ContentRecommender()
        recommender.initialize_vectorstore()
        
        recs = recommender.get_recommendations("sad", "User had a bad day")
        print(f"✅ Got {len(recs)} recommendations")
        print(f"   First: {recs[0]['title']}\n")
    except Exception as e:
        print(f"❌ Recommender Error: {e}\n")
    
    # Test 3: Grok Recommendations
    print("3️⃣ Testing Grok Recommendations...")
    try:
        insight = grok.get_recommendation(
            emotion="happy",
            context="User just got promoted",
            available_content=recs[:3]
        )
        print(f"✅ Grok Insight: {insight[:200]}...\n")
    except Exception as e:
        print(f"❌ Insight Error: {e}\n")
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_all()
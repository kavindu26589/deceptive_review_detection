import requests

API_URL = "http://127.0.0.1:8000/analyze"

TEST_DATA = [
    {
        "id": 1,
        "text": "This laptop is incredibly fast. Boot time is under 10 seconds. However, I find myself waiting 5 minutes just to open Chrome. The performance is unmatched in this price range."
    },
    {
        "id": 2,
        "text": "The camera quality is stunning in daylight. Night mode works well too. I've taken beautiful photos at my daughter's evening recital. Great for any lighting condition."
    },
    {
        "id": 3,
        "text": "I've never had a phone this durable. Dropped it multiple times with no damage. The screen cracked on the first drop though. Build quality is exceptional."
    },
    {
        "id": 4,
        "text": "Customer service was unhelpful and rude. They resolved my issue within minutes and even gave me a discount. Worst support experience I've ever had."
    },
    {
        "id": 5,
        "text": "The noise cancellation is mediocre at best. I can still hear my coworkers clearly. But honestly, for the price, you can't expect studio-quality isolation."
    },
    {
        "id": 6,
        "text": "Shipping was lightning fast - arrived in 2 days. The three-week wait was worth it though. Amazon Prime really delivers."
    },
    {
        "id": 7,
        "text": "This blender is whisper quiet. My baby sleeps right through it. The noise is so loud I have to wear ear protection. Perfect for early morning smoothies."
    },
    {
        "id": 8,
        "text": "Not the cheapest option, but definitely worth the premium price. The quality justifies the cost. You get what you pay for with this brand."
    }
]

def main():
    print("\n🚀 Running Deceptive Review Detection Tests\n")

    for item in TEST_DATA:
        print("=" * 80)
        print(f"Review ID: {item['id']}")
        print("-" * 80)
        print(item["text"])
        print("\nResult:")

        response = requests.post(API_URL, json={"text": item["text"]})
        result = response.json()

        if result["has_contradiction"]:
            print("❌ Contradiction: YES")
            print(f"Confidence: {result['confidence']:.2f}")
        else:
            print("✅ Contradiction: NO")
            print(f"Confidence: {result['confidence']:.2f}")

        print("Explanation:")
        print(result["explanation"])

        if result["pairs"]:
            print("\nContradicting Pairs:")
            for a, b in result["pairs"]:
                print(f" - A: {a}")
                print(f"   B: {b}")

    print("\n✅ All tests completed.\n")

if __name__ == "__main__":
    main()

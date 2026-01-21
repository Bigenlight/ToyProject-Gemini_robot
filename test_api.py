# test_api.py 수정본
from google import genai
from google.genai import types

MY_API_KEY = "AIzaSyBpBo1uoFaiwht8jy5VmwVxpCf11aE3bzg"
client = genai.Client(api_key=MY_API_KEY)

def test_gemini():
    try:
        print("🤖 [System] Gemini에게 질문을 던지는 중...")
        
        # 안전 필터를 끄고 다시 시도 (테스트용)
        config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )

        response = client.models.generate_content(
            model="gemini-3-flash-preview", # 3-flash-preview도 가능
            contents="안녕! 너는 로봇 팔 제어 에이전트야. 준비됐니?",
            config=config
        )
        
        # 1. 응답 텍스트 확인
        if response.text:
            print(f"\n🧠 [Gemini]: {response.text}")
        else:
            # 2. 텍스트가 없다면 차단 사유 확인
            print("\n⚠️ [Warning]: 텍스트 응답이 없습니다.")
            print(f"차단 사유(Prompt Feedback): {response.prompt_feedback}")
            print(f"후보 응답 확인: {response.candidates[0].finish_reason}")

    except Exception as e:
        print(f"❌ [Error] 상세 오류 발생: {e}")

if __name__ == "__main__":
    test_gemini()
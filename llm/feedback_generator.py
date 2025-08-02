import openai

client = openai.OpenAI()  # uses OPENAI_API_KEY from env by default

def generate_feedback(prediction, confidence=None, history=None):
    """
    Generates personalized GPT feedback based on brain model output.
    """
    label = "left-hand" if prediction == 0 else "right-hand"

    prompt = f"""
    You are a neurofeedback coach. The user just completed a motor imagery trial.
    They imagined moving their {label}. 

    The brain-signal classifier predicted this with {'high' if confidence and confidence > 0.8 else 'low' if confidence else 'unknown'} confidence.

    {f"Recent trials showed signs of fatigue or misalignment." if history and history[-1] == 'low' else ""}

    Give the user short, motivational, and actionable feedback.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a kind, motivational brain coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


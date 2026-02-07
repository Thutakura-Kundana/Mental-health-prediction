# utils/recommender.py

def get_recommendations(stress_level, confidence=1.0):
    """
    Returns a list of personalized, professionally worded recommendations
    based on stress level and model confidence.

    Parameters:
    - stress_level (int): Predicted stress level (e.g., 0 = not stressed, 1 = stressed)
    - confidence (float): Model confidence score (optional, default = 1.0)

    Returns:
    - List of recommended actions (List[str])
    """
    if stress_level == 1 and confidence > 0.8:
        return [
            "Engage in a 5-minute guided diaphragmatic breathing session to reduce acute stress.",
            "Listen to a curated playlist of calming instrumental music to regulate heart rate variability.",
            "Practice progressive muscle relaxation to alleviate physical tension and promote calm.",
            "Disconnect briefly from digital devices and seek a low-stimulation environment.",
            "Consider a short mindfulness meditation using a trusted app like Headspace or Calm."
        ]
    elif stress_level == 1:
        return [
            "Take a 15-minute walk outdoors to stimulate endorphin release and reduce cortisol levels.",
            "Write down three positive experiences from today to encourage cognitive reframing.",
            "Perform a brief gratitude journaling exercise to shift attention from stressors.",
            "Hydrate and consume a light, balanced snack to stabilize energy and mood.",
            "Engage in a short stretching routine to improve circulation and reduce muscular tension."
        ]
    else:
        return [
            "Maintain your current wellness routine—consistency is key to long-term resilience.",
            "Reflect on recent coping strategies that worked well and document them for future use.",
            "Schedule a preventive check-in with a mental health coach or peer support group.",
            "Explore a new hobby or creative outlet to reinforce psychological flexibility.",
            "Practice a short visualization exercise to reinforce a sense of calm and control."
        ]
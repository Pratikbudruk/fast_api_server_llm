from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import re

# Load the HuggingFace model pipeline (text generation)
model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=-1)  # device=-1 for CPU

app = FastAPI()


class PatientDataRequest(BaseModel):
    text: str


# Function to calculate BMI
def calculate_bmi(weight, height):
    if weight and height:
        height_m = height / 100  # Convert cm to meters
        return round(weight / (height_m ** 2), 1)
    return None


def extract_patient_data(text: str):
    prompt = f"""
    Extract the following information from the given patient description:
    - Name 
    - Gender 
    - Age 
    - Weight (kg) 
    - Height (cm) 
    - BMI 
    - Chief Medical Complaint

    Return the result as a Python dictionary. If some fields are missing, set their values to None.

    Input: "{text}"
    Output:
    """
    response = model(prompt, max_length=300, num_return_sequences=1)[0]["generated_text"]

    try:
        result = eval(re.search(r'\{.*\}', response).group())  # Extract dictionary-like text
        if "weight" in result and "height" in result:
            result["BMI"] = calculate_bmi(result.get("weight"), result.get("height"))
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse response: {e}")


@app.post("/extract_patient_data/")
async def get_patient_data(request: PatientDataRequest):
    try:
        # Extract structured data from the text
        result = extract_patient_data(request.text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

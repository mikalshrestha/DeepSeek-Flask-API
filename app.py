from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
HF_AUTH_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_AUTH_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, token=HF_AUTH_TOKEN, torch_dtype=torch.float16
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    input_text = data.get("text", "")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model.to(device, dtype=dtype)
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Check for invalid values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return jsonify({"error": "Model generated NaN or Inf values"}), 500

        logits = logits - logits.max()  # Normalize logits

    output = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

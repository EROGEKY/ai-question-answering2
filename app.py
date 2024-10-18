from flask import Flask, request, jsonify
from transformers import pipeline

# إعداد Flask API
app = Flask(__name__)

# تحميل نموذج السؤال والإجابة من Hugging Face
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# API لاستقبال السؤال والسياق
@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.get_json()
    user_question = data.get('question', '')
    context = data.get('context', '')

    # استخدام النموذج للإجابة على السؤال
    result = qa_pipeline(question=user_question, context=context)

    return jsonify({"answer": result['answer']}), 200

# تشغيل الخادم
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
